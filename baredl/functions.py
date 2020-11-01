import numpy as np
from .core import Tensor, Function, reverse_broadcast_to, get_array_module
from .utils import logsumexp, pair, im2col_array, col2im_array, get_deconv_outsize


# -------------------------------------------------------------
# Basic functions: exp / log / sin / cos / tanh
# -------------------------------------------------------------


class Exp(Function):
    def forward(self, x):
        """
        Parameters
        ----------
        x: xp.ndarray (baredl.Tensor.data)

        Returns
        -------
        y: xp.ndarray
        """
        xp = get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy):
        """
        Parameters
        ----------
        gy: baredl.Tensor (baredl.Tensor.grad)

        Returns
        -------
        gx: baredl.Tensor
        """
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


class LeakyReLU(Function):
    def __init__(self, slope):
        self.slope = slope

    def forward(self, x):
        y = x.copy()
        y[x <= 0] *= self.slope
        return y

    def backward(self, gy):
        x, = self.inputs 
        mask = (x.data > 0).astype(gy.dtype)
        mask[mask <= 0] = self.slope
        gx = gy * mask
        return gx


def leaky_relu(x, slope=0.2):
    return LeakyReLU(slope)(x)


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
        gb = None if b.data is None else reverse_broadcast_to(gy, b.shape)
        gx = gy @ W.T # gx = matmul(gy, W.T)
        gW = x.T @ gy # gW = matmul(x.T, gy)
        return gx, gW, gb


def linear(x, W, b=None):
    return Linear()(x, W, b)


class Dropout(Function):
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio

    def forward(self, x, training=True):
        if training:
            xp = get_array_module(x)
            self.mask = xp.random.rand(*x.shape) > self.dropout_ratio
            self.scale = xp.array(1.0 - self.dropout_ratio).astype(x.dtype)
            y = x * self.mask / self.scale
            return y
        else:
            self.mask, self.scale = 1.0, 1.0 # practically no need to do this though.
            return x

    def backward(self, gy):
        gx = gy * self.mask / self.scale
        return gx


def dropout(x, dropout_ratio=0.5, training=True):
    return Dropout(dropout_ratio)(x, training)


# -------------------------------------------------------------
# Conv functions: 
# -------------------------------------------------------------


class Conv2d(Function):

    def __init__(self, stride=1, padding=0):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(padding)

    def forward(self, x, W, b):
        """
        Parameters
        ----------
        x: xp.ndarray (N, C, H, Width)
            N: number of samples (images)
            C: number of channels
            H: height of the images
            Width: width of the images
            e.g. x with shape (3,2,3,4) is like below.
                np.array([
                    # sample1
                    [
                        [[0,0,0,0],[0,0,0,0],[0,0,0,0]], # channel1 (3*4 matrix)
                        [[0,0,0,0],[0,0,0,0],[0,0,0,0]]  # channel2 (3*4 matrix)
                    ],
                    # sample2
                    [
                        [[0,0,0,0],[0,0,0,0],[0,0,0,0]], # channel1 (3*4 matrix)
                        [[0,0,0,0],[0,0,0,0],[0,0,0,0]]  # channel2 (3*4 matrix)
                    ],
                    # sample 3
                    [
                        [[0,0,0,0],[0,0,0,0],[0,0,0,0]], # channel1 (3*4 matrix)
                        [[0,0,0,0],[0,0,0,0],[0,0,0,0]]  # channel2 (3*4 matrix)
                    ],
                ])

        W: xp.ndarray (OC, C, KH, KW)
            OC: number of output channels
            C: number of channels 
            KH: height of the kernel (filter)
            KW: width of the kernel (filter)
            e.g. W with shape (4,2,2,2) is like below.
                np.array([
                    # output channel 1
                    [
                        [[0,0],[0,0]], # channel1 (2*2 kernel)
                        [[0,0],[0,0]]  # channel2 (2*2 kernel)
                    ],
                    # output channel 2
                    [
                        [[0,0],[0,0]], # channel1 (2*2 kernel)
                        [[0,0],[0,0]]  # channel2 (2*2 kernel)
                    ],
                    # output channel 3
                    [
                        [[0,0],[0,0]], # channel1 (2*2 kernel)
                        [[0,0],[0,0]]  # channel2 (2*2 kernel)
                    ],
                    # output channel 4
                    [
                        [[0,0],[0,0]], # channel1 (2*2 kernel)
                        [[0,0],[0,0]  # channel2 (2*2 kernel)
                    ],
                ])

        b: xp.ndarray (OC,)
            OC: number of output channels
            e.g. b with shape (4,) is like below.
                np.array([0, 0, 0, 0]) # each 0 is for each output channel

        Returns
        -------
        y: xp.ndarray (N, OC, OH, OW)
            N: number of samples (images)
            OC: number of output channels
            OH: height of output images
            OW: width of output images
            e.g. with all the examples above, y's shape would be (3,4,2,3) like below.
                np.array([
                    # sample1
                    [
                        [[0,0,0],[0,0,0]], # output channel1 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel2 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel3 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel4 (2*3 matrix)
                    ],
                    # sample2
                    [
                        [[0,0,0],[0,0,0]], # output channel1 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel2 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel3 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel4 (2*3 matrix)
                    ],
                    # sample 3
                    [
                        [[0,0,0],[0,0,0]], # output channel1 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel2 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel3 (2*3 matrix)
                        [[0,0,0],[0,0,0]], # output channel4 (2*3 matrix)
                    ],
                ])
        """
        xp = get_array_module(x)

        KH, KW = W.shape[2:]

        # col is a xp.ndarray (N, C, KH, KW, OH, OW)
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        # col is a xp.ndarray (N,  C, KH, KW, OH, OW)
        # W   is a xp.ndarray (OC, C, KH, KW)
        # y   is a xp.ndarray (N, OH, OW, OC)
        axes = ((1, 2, 3), (1, 2, 3))
        y = xp.tensordot(col, W, axes=axes)

        if b is not None:
            # y is a xp.ndarray (N, OH, OW, OC)
            # b is a xp.ndarray (OC,)
            # here, b is broadcasted to (1, 1, 1, OC) then (N, OH, OW, OC)
            y += b

        # (N, OH, OW, OC) to (N, OC, OH, OW)
        y = xp.rollaxis(y, 3, 1)

        return y

    def backward(self, gy):
        """
        Parameters
        ----------
        gy: baredl.Tensor (N, OC, OH, OW)
            forward's output's grad.

        Returns 
        -------
        gx: baredl.Tensor (N, C, H, Width)

        gW: baredl.Tensor (OC, C, KH, KW)
        
        gb: baredl.Tensor (OC,)
        """
        x, W, b = self.inputs

        H, Width = x.shape[2], x.shape[3]
        
        gx = conv_transpose2d(gy, W, b=None, stride=self.stride, padding=self.pad, outsize=(H, Width))
        gW = Conv2dGradW(self)(x, gy)
        gb = None

        # note b (input) is stored as a Tensor even if it's None.
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))

        return gx, gW, gb


def conv2d(x, W, b=None, stride=1, padding=0):
    return Conv2d(stride, padding)(x, W, b)


class ConvTranspose2d(Function):
    
    def __init__(self, stride=1, padding=0, outsize=None):
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(padding)
        self.outsize = outsize

    def forward(self, x, W, b):
        """
        Parameters
        ----------
        x: xp.ndarray (N, C, H, Width)
            N: number of samples (images)
            C: number of channels
            H: height of the images
            Width: width of the images
        
        W: xp.ndarray (C, OC, KH, KW)
            C: number of channels
            OC: number of output channels
            KH: height of the kernel (filter)
            KW: width of the kernel (filter)
            Note that shape of the input W in Conv2d is (OC, C, KH, KW)
            whereas this input W is (C, OC, KH, KW). This is because Conv2d's 
            input channel C is the output channel OC from the perspective of ConvTranspose2d.
            And Conv2d's output chanel OC is the input chanel C of the ConvTranspose2d.
            So, same W, but the way we call C, OC is opposite from the perspective of Conv2d 
            vs ConvTranspose2d.

        b: xp.ndarray (OC,)
            OC: number of output channels
            e.g. b with shape (4,) is like below.
                np.array([0, 0, 0, 0]) # each 0 is for each output channel

        Returns
        -------
        y: xp.ndarray (N, OC, OH, OW)
        """
        xp = get_array_module(x)

        Weight = W
        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = Weight.shape
        N, C, H, W = x.shape 

        if self.outsize is None:
            OH = get_deconv_outsize(H, KH, SH, PH)
            OW = get_deconv_outsize(W, KW, SW, PW)
        else:
            OH, OW = pair(self.outsize)

        # Note that ConvTranspose2d is reverse operation of Conv2d.
        # So, below (N, OC, OH, OW) is Conv2d's (N, C, H, W)
        img_shape = (N, OC, OH, OW)

        # Weight's shape is (C, OC, KH, KW)
        # x's shape is      (N, C,  H,  W)
        # gcol's shape is   (OC, KH, KW, N, H, W)
        gcol = xp.tensordot(Weight, x, (0,1))

        # (OC, KH, KW, N, H, W) to (N, OC, KH, KW, H, W)
        gcol = xp.rollaxis(gcol, 3)

        # y's shape is (N, OC, OH, OW)
        y = col2im_array(gcol, img_shape, (KH, KW), self.stride, self.pad,
                         to_matrix=False)

        if b is not None:
            # b's shape is (OC,)
            # since (OC,) cannot be broadcasted to (N, OC, OH, OW),
            # need to respahe to (1, OC, 1, 1) first. Then 
            # (1, OC, 1, 1) will be broadcasted to (N, OC, OH, OW)
            y += b.reshape((1, b.size, 1, 1))

        return y

    def backward(self, gy):
        """
        Parameters
        ----------
        gy: baredl.Tensor (N, OC, OH, OW)
            forward's output's grad.

        Returns 
        -------
        gx: baredl.Tensor (N, C, H, Width)

        gW: baredl.Tensor (C, OC, KH, KW)
        
        gb: baredl.Tensor (OC,)
        """
        x, W, b = self.inputs

        gx = conv2d(gy, W, b=None, stride=self.stride, padding=self.pad)
        gW = Conv2dGradW(self)(gy, x) # not (x, gy, self) but (gy, x, self)
        gb = None

        # note b (input) is stored as a Tensor even if it's None.
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))

        return gx, gW, gb


def conv_transpose2d(x, W, b=None, stride=1, padding=0, outsize=None):
    return ConvTranspose2d(stride, padding, outsize)(x, W, b)

        
class Conv2dGradW(Function):
    def __init__(self, conv2d):
        W = conv2d.inputs[1]
        KH, KW = W.shape[2:]
        self.kernel_size = (KH, kW)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x, gy):
        xp = get_array_module(x)

        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        gW = xp.tensordot(gy, col, ((0,2,3),(0,4,5)))
        return gW

    def backward(self, gys):
        x, gy = self.inputs
        gW, = self.outputs

        XH, XW = x.shape[2:]
        gx = conv_transpose2d(gy, gW, stride=self.stride, padding=self.pad,outsize=(XH, XW))
        ggy = conv2d(x, gW, stride=self.stride, pad=self.pad)
        return gx, ggy


# -------------------------------------------------------------
# Pooling functions: 
# -------------------------------------------------------------


class MaxPool2d(Function):
    def __init__(self, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = padding

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        y = col.max(axis=2)
        self.indices = col.argmax(axis=2)

        return y

    def backward(self, gy):
        return MaxPool2dGrad(self)(gy)


def max_pool2d(x, kernel_size, stride=1, padding=0):
    return MaxPool2d(kernel_size, stride, padding)(x)


class MaxPool2dGrad(Function):
    def __init__(self, maxpool2d):
        self.maxpool2d = maxpool2d
        self.kernel_size = maxpool2d.kernel_size
        self.stride = maxpool2d.stride
        self.pad = maxpool2d.pad
        self.input_shape = maxpool2d.inputs[0].shape
        self.dtype = maxpool2d.inputs[0].dtype
        self.indices = maxpool2d.indices

    def forward(self, gy):
        xp = get_array_module(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        gcol = xp.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        indices = (self.indices.ravel() + xp.arange(0, self.indices.size * KH * KW, KH * KW))

        gcol[indices] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        gx = col2im_array(gcol, (N, C, H, W), self.kernel_size, self.stride, self.pad, to_matrix=False)

        return gx

    def backward(self, ggx):
        f = MaxPool2dWithIndices(self.maxpool2d)
        return f(ggx)


class MaxPool2dWithIndices(Function):

    def __init__(self, maxpool2d):
        self.kernel_size = maxpool2d.kernel_size
        self.stride = maxpool2d.stride
        self.pad = maxpool2d.pad
        self.input_shape = maxpool2d.inputs[0].shape
        self.dtype = maxpool2d.inputs[0].dtype
        self.indices = maxpool2d.indices

    def forward(self, x):
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indices = self.indices.ravel()
        col = col[np.arange(len(indices)), indices]
        return col.reshape(N, C, OH, OW)


        
