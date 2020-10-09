import contextlib
import weakref
import numpy as np

import deeplfs

##############################
# Configuration
##############################

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)


##############################
# Utils
##############################

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

##############################
# Variable / Parameter / Base function
##############################

class Variable:

    __array_priority__ = 200  
    # to prioritise __radd__ of this class over np.ndarray's __add__
    # when np.array([2.0]) * Variable(np.array([1.0]))
    
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'variable(' + p + ')'

    def __getitem__(self, slices):
        return get_item(self, slices)

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(self, other)
    
    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(self, other)

    def __neg__(self):
        return neg(self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return rsub(self, other)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return rdiv(self, other)

    def __pow__(self, other):
        return pow(self, other)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return deeplfs.functions.transpose(self)

    def transpose(self):
        return deeplfs.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return deeplfs.functions.sum(self, axis, keepdims)

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)
            
            if not retain_grad: # Note: even not retain_grad, grad of end nodes will be still retained
                for y in f.outputs:
                    y().grad = None

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return deeplfs.functions.reshape(self, shape)

class Parameter(Variable):
    pass

class Function:
    def __call__(self, *inputs):

        inputs = [as_variable(input) for input in inputs]

        xs = [input.data for input in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple): # ここちょい気持ち悪いかも
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([input.generation for input in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

##############################
# Basic arithmetic operations
##############################

class Add(Function):
    def forward(self, x0, x1):
        """ x0, x1: np.ndarray (any shape) """
        y = x0 + x1 # if x0 and x1 have different shape, np automatically broadcast.
        return y 
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        # for broadcast
        x0, x1 = self.inputs
        if x0.shape != x1.shape:
            gx0 = deeplfs.functions.sum_to(gx0, x0.shape)
            gx1 = deeplfs.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Mul(Function):
    def forward(self, x0, x1):
        """ x0, x1: np.ndarray (any shape) """
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            gx0 = deeplfs.functions.sum_to(gx0, x0.shape)
            gx1 = deeplfs.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        # for broadcast
        x0, x1 = self.inputs
        if x0.shape != x1.shape:
            gx0 = deeplfs.functions.sum_to(gx0, x0.shape)
            gx1 = deeplfs.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x0
        gx1 = gy * (- x0 / (x1 ** 2))
        if x0.shape != x1.shape:
            gx0 = deeplfs.functions.sum_to(gx0, x0.shape)
            gx1 = deeplfs.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = self.c * (x ** (self.c - 1)) * gy
        return gx

def add(x0, x1):
    """ 
    x0, x1: np.ndarray (any shape) or scalar 
    Since this function is only used to override Variable class's 
    __add__ and __radd__, 
    input x0 is always a Variable's data. Which means always np.ndarray.
    In contrast, x1 can be a scalar. e.g. Variable(np.array(1)) + 3.0
    """
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    """ 
    x0, x1: np.ndarray (any shape) or scalar 
    Since this function is only used to override Variable class's 
    __mul__ and __rmul__, 
    input x0 is always a Variable's data. Which means always np.ndarray.
    In contrast, x1 can be a scalar. e.g. Variable(np.array(1)) * 3.0
    """
    x1 = as_array(x1)
    return Mul()(x0, x1)

def neg(x):
    """ 
    x: np.ndarray (any shape)
    Since this function is only used to override Variable class's 
    __neg__, 
    input x is always a Variable's data. Which means always np.ndarray.
    """
    return Neg()(x)

def sub(x0, x1):
    """ 
    x0, x1: np.ndarray (any shape) or scalar 
    Since this function is only used to override Variable class's 
    __sub__, 
    input x0 is always a Variable's data. Which means always np.ndarray.
    In contrast, x1 can be a scalar. e.g. Variable(np.array(1)) - 3.0
    """
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    """ 
    x0, x1: np.ndarray (any shape) or scalar 
    Since this function is only used to override Variable class's 
    __rsub__, 
    input x0 is always a Variable's data. Which means always np.ndarray.
    In contrast, x1 can be a scalar. e.g. 3.0 - Variable(np.array(1))
    """
    x1 = as_array(x1)
    return Sub()(x1, x0)

def div(x0, x1):
    """ 
    x0, x1: np.ndarray (any shape) or scalar 
    """
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    """ 
    x0, x1: np.ndarray (any shape) or scalar 
    """
    x1 = as_array(x1)
    return Div()(x1, x0)

def pow(x,c):
    """ 
    x: np.ndarray (any shape) or scalar 
    """
    return Pow(c)(x)


##############################
# get_item operation
##############################

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)

def get_item(x, slices):
    return GetItem(slices)(x)

class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        gx = np.zeros_like(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


##############################
# Layer and Model
##############################

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

class Model(Layer):
    pass

##############################
# Optimizer
##############################

class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        # list parameters containing not None grad
        params = [p for p in self.target.params() if p.grad is not None]

        # preprocess (optional)
        for f in self.hooks:
            f(params)

        for p in params:
            self.update_one(p)

    def update_one(param):
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)

    