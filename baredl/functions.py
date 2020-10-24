import numpy as np
from baredl.core import Variable, Function, as_variable, sum_to, get_array_module
from baredl.config import Config
from baredl.utils import logsumexp


# -------------------------------------------------------------
# Basic functions: exp / log / sin / cos / tanh
# -------------------------------------------------------------


class Exp(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        #x = self.inputs[0].data
        #gx = np.exp(x) * gy
        y = self.outputs[0]() # weakref
        gx = gy * y
        return gx


def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx


def log(x):
    return Log()(x)


class Sin(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]() # weakref
        gx = gy * (1 - y ** 2)
        return gx


def tanh(x):
    return Tanh()(x)


# -------------------------------------------------------------
# Activation functions: sigmoid / softmax
# -------------------------------------------------------------


class Sigmoid(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = 1 / (1 + xp.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1-y)
        return gx


def sigmoid(x):
    return Sigmoid()(x)


class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]() # weakref
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x, axis=1):
    return Softmax(axis)(x)


class ReLU(Function):
    def forward(self, x):
        xp = get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs 
        mask = x.data > 0 
        gx = gy * mask
        return gx


def relu(x):
    return ReLU()(x)


# -------------------------------------------------------------
# Loss functions: mean_squared_error
# -------------------------------------------------------------


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2) / len(diff)
        return y 
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        log_z = logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs 
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)
        xp = get_array_module(t.data)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)


# -------------------------------------------------------------
# Other functions: linear, dropout
# -------------------------------------------------------------


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = gy @ W.T # gx = matmul(gy, W.T)
        gW = x.T @ gy # gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


class Dropout(Function):
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def forward(self, x):
        if Config.is_train:
            xp = get_array_module(x)
            self.mask = xp.random.rand(*x.shape) > self.dropout_ratio
            self.scale = xp.array(1.0 - self.dropout_ratio).astype(x.dtype)
            y = x * self.mask / self.scale
            return y
        else:
            return x

    def backward(self, gy):
        if Config.is_train:
            gx = gy * self.mask / self.scale
            return gx
        else:
            return gy


def dropout(x, dropout_ratio=0.5):
    return Dropout(dropout_ratio)(x)