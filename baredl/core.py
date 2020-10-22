import contextlib
import weakref
import numpy as np

import baredl

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
    """
    Convert scalar x to np.array datatype.
    e.g. 3 -> np.array(3)

    Parameters
    ----------
    x: np.ndarray (any shape), np.scalar or scalar
    """
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    """
    Convert np.array object to Variable.
    e.g. np.array([1,2]) -> Variable([1,2])

    Parameters
    ----------
    obj: np.ndarray (any shape) of real (-inf, inf)
    """
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

##############################
# Variable / Parameter 
##############################

class Variable:
    """
    Data container class

    Parameters
    ----------
    data: np.ndarray (any shape) of real (-inf, inf)
    name: string
    """

    # to prioritise __radd__ of this class over np.ndarray's __add__
    # when np.array([2.0]) + Variable(np.array([1.0]))
    # also same for __rmul__ vs np.ndarray's __mul__
    __array_priority__ = 200  
    
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None # function which generated this Variable instance
        # generation indicates "depth" of the position of this variable 
        # in the calculation graph. This is important when we perform
        # backprop in a complex calculation graph.
        self.generation = 0

    def __len__(self):
        """ define len(Variable) """
        return len(self.data)

    def __repr__(self):
        """ define print(Variable) """
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9) # 9 is length of "variable("
        return 'variable(' + p + ')'

    def __getitem__(self, slices):
        """ 
        define Variable[...] 
        e.g. Variable[:,2] , Variable[1,1], Variable[[0,1,1]]
        """
        return get_item(self, slices)

    def __add__(self, other):
        """ define Variable +  """
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
        return baredl.functions.transpose(self)

    def transpose(self):
        return baredl.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return baredl.functions.sum(self, axis, keepdims)

    def set_creator(self, func):
        self.creator = func
        # generation of this Variable instance will be 1 step deeper 
        # than the function created this instance. 
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        """
        calculate gradients of ancestor Variable instances by backprop.

        Parameters
        ----------
        retain_grad: bool
            If True, keep grad values of every single Variable instance
            in the calculation graph. 
            If False, only keep grad values of "end node" Variable instances.
            This is for memory efficiency purpose. In most cases, False is fine. 
        
        create_graph: bool
            This indicates if we need to keep calculation graph for gradients. 
            if True, we keep calculation graph for grad i.e. grad.backward() is available.
            This needs to be True only if you need to do double backprop. 
        """
        
        # "self.grad is None" means this Variable is the starting point
        # of the backprop. Because if this Variable instance is in the 
        # middle of backprop, self.grad should be already defined (not None)
        # by the time this backward is called. 
        # Init value is always 1 because, e.g. forward flow is "x -> z -> L"
        # then backprop is dL/dx = dL/dL * dL/dz * dz/dx 
        # where the starting point dL/dL is always 1. 
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data)) # grad is also a Variable!! This allows us double backprop.

        # funcs is a list to store Function instances of which 
        # backward need to be called.
        funcs = []
        # seen_set is a set to store Function instances which 
        # we ran backward once. 
        # this is to prevent same Function instance's backward
        # is called and calculated multiple times by mistake. 
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)

                # sort Function instances in funcs by generation.
                # so that always Function instances in "deeper"
                # position's called first.
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop() # since funcs is sorted by generation, this always gives the func with the largest generation.
            
            # gradients of all outputs of f. The order of elements is corresponding to the order of ys = f.forward(*xs).
            # Note: we access like "output()"" not just "output" because it's a weakref.
            gys = [output().grad for output in f.outputs]

            # if create_graph is False (which is most likely), 
            # do not keep calculation graph for grad calculation.
            with using_config('enable_backprop', create_graph):

                # calculate the gradients of f's inputs Variable instances
                # using gradients of f's outputs.
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple): # make sure gxs is a tuple format
                    gxs = (gxs,)
                
                # set f's input's gradient
                for x, gx in zip(f.inputs, gxs): # Note: order of f.inputs & order of gxs = f.backward(*gys) is corresponding. so zip works. 
                    if x.grad is None:
                        x.grad = gx
                    else:
                        # this is the case when f's input x is also an input of 
                        # another Function instance, and already grad given by its backward.
                        # in that case, we add gradient from this f on top of it. 
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)
            
            if not retain_grad: # Note: even not retain_grad, grad of end nodes will be still retained
                for y in f.outputs:
                    y().grad = None

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return baredl.functions.reshape(self, shape)


class Parameter(Variable):
    pass


##############################
# Base Functions
##############################


class Function:
    """
    Base class of all functions defined in baredl, which operate/manipulate Variable instances.
    Functions can take np.ndarray as a input but it will be converted into Variable. 
    """

    def __call__(self, *inputs):
        """
        Perform the operation (specified in self.forward) on the data of the given
        Variable instances. Return the result as a (list of) Variable instance(s).

        Parameters
        ----------
        inputs: a tuple of one or more of Variable or np.ndarray (any shape) of real (-inf, inf)

        Returns
        -------
        Outputs: a list of Variable, or a Variable
        """

        # make sure every input is Variable datatype
        inputs = [as_variable(input) for input in inputs]

        # take data (np.ndarray) from each input (Variable)
        xs = [input.data for input in inputs] # xs: list of np.ndarray

        # perform operation on the data (np.ndarray)
        ys = self.forward(*xs) # ys: np.ndarray or tuple of np.ndarray
        if not isinstance(ys, tuple): # if ys is a np.ndarray, convert to a tuple
            ys = (ys,)

        # each element of the tuple ys is most likely to be a np.ndarray
        # but in case of it's a scalar, apply as_array() and then make it as a Variable.
        outputs = [Variable(as_array(y)) for y in ys]

        # Keeping references to inputs / outputs are for backprop purpose. 
        # This is always needed at training, but no need at inference (prediction). 
        # So when we call this at inference, turn off this block to reduce memory usage, using Config.
        # Also, when this Function instance is called in backward (i.e. calculation of gradient), 
        # we most likely don't need to store these information unless we need to do double backprop.
        if Config.enable_backprop:
            # self.generation value indicates how deep this function is in the entire calc graph.
            # in the other words, how far this function is from the first input,
            # or how close this function is from the final outputs of the calc graph.
            # set this Function instance's generation as same as the biggest generation out of 
            # its inputs. 
            # this is because we want to make sure to run backward of Function instances in
            # deeper places before running backward of Function instances in less deep place 
            # in the calc graph.
            self.generation = max([input.generation for input in inputs])

            # set all outputs' creator as this Function instance.
            # so that those outputs can refer to this when backprop. 
            for output in outputs:
                output.set_creator(self)

            # store the reference to the inputs, so that 
            # this Function instance can refer to them when backprop.
            self.inputs = inputs

            # also needs to store the reference to the outputs,
            # as we use outputs' grads for self.backward().
            # however, as outputs have also reference to this Function instance.
            # to prevent a circular reference issue, we use weakref.
            self.outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        """ x should be one or more of np.ndarray (input Variable's data) """
        raise NotImplementedError()

    def backward(self, gy):
        """ gy should be one or more of Variable (output Variable's grad) """
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
            gx0 = baredl.functions.sum_to(gx0, x0.shape)
            gx1 = baredl.functions.sum_to(gx1, x1.shape)
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
            gx0 = baredl.functions.sum_to(gx0, x0.shape)
            gx1 = baredl.functions.sum_to(gx1, x1.shape)
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
            gx0 = baredl.functions.sum_to(gx0, x0.shape)
            gx1 = baredl.functions.sum_to(gx1, x1.shape)
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
            gx0 = baredl.functions.sum_to(gx0, x0.shape)
            gx1 = baredl.functions.sum_to(gx1, x1.shape)
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

