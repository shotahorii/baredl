import numpy as np
from baredl.core import Variable, Function, as_variable

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


"""
Note about input of below function classes
This is how forward methods are called: (in Function class)
---
xs = [input.data for input in inputs]
ys = self.forward(*xs)
---
So, x in forward method is a tuple of np.ndarray

This is how backward methods are called: (in Variable class)
---
gys = [output().grad for output in f.outputs]
with using_config('enable_backprop', create_graph):
    gxs = f.backward(*gys)
---
So, gy in backward method is a tuple of Variable
"""


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx

def log(x):
    return Log()(x)


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y ** 2)
        return gx

def tanh(x):
    return Tanh()(x)


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes==None:
            return tanspose(gy)

        inv_axes = tuple(np.argsort(self.axes)) # should I use tuple(np.argsort([ax % len(self.axes) for ax in self.axes])) ??
        return transpose(x, inv_axes)

def transpose(x, axes=None):
    return Transpose(axes)(x)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        
        # Here, we'd like to reshape gy back to original x shape
        # by using broadcast_to.
        # To do so, we firstly want to make sure input gy has  
        # the shape which is broadcast-able to original x shape.
        # What does this mean? For example, 
        # x = Variable(np.array([[1,2,3],[4,5,6]]))
        # y = sum(x, axis=1, keepdims=False)
        # then, y is Variable([6, 15])
        # hence, gy is Variable([1, 1])
        # now, x.shape is (2,3)
        # Variable([1, 1]) cannot be broadcasted to Variable([[1,1,1],[1,1,1]])
        # i.e. gy needs to be Variable([[1], [1]]) instead of Variable([1, 1])        
        gy = self._reshape_broadcastable(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx

    def _reshape_broadcastable(self, gy, x_shape, axis, keepdims):
        """Reshape gradient appropriately for dezero.functions.sum's backward.
        Args:
            gy (dezero.Variable): Gradient variable from the output by backprop.
            x_shape (tuple): Shape used at sum function's forward.
            axis (None or int or tuple of ints): Axis used at sum function's
                forward.
            keepdims (bool): Keepdims used at sum function's forward.
        Returns:
            dezero.Variable: Gradient variable which is reshaped appropriately
        """
        ndim = len(x_shape)

        # standardise axis
        if axis is None:
            tupled_axis = None
        elif not isinstance(axis, tuple): # only one axis e.g. axis=0
            tupled_axis = (axis,)
        else: # multiple axes e.g. axis=(0,1)
            tupled_axis = axis

        if ndim == 0 or tupled_axis is None or keepdims:
            # if ndim==0, x is a scalar, then y and gy are also scalar.
            # hence no problem with broadchasting.
            # if tupled_axis is None, y and gy are always scalar. 
            # hence no problem with broadcasting.
            # if keepdims==True, shape after sum is kept as same. 
            # so no need to reshape to broadcast. 
            return gy
        else:
            actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis] # just deal with negative axes
            shape = list(gy.shape)
            for ax in sorted(actual_axis): # fill the summed dimentions with 1. So that broadcastable.
                shape.insert(ax, 1)
            
            return gy.reshape(shape) 

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = self._sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx

    def _sum_to(self, x, shape):
        """Sum elements along axes to output an array of a given shape.
        Args:
            x (ndarray): Input array.
            shape:
        Returns:
            ndarray: Output array of the shape.
        """
        ndim = len(shape)
        lead = x.ndim - ndim
        lead_axis = tuple(range(lead))

        axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
        y = x.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
            y = y.squeeze(lead_axis)
        return y

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW
    
def matmul(x, W):
    return MatMul()(x, W)


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)


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


class Sigmoid(Function):
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1-y)
        return gx

def sigmoid(x):
    return Sigmoid()(x)