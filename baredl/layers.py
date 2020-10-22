import weakref
import numpy as np
from baredl.core import Parameter
import baredl.functions as F

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


class Linear(Layer):
    def __init__(self, out_size, in_size=None, nobias=False, dtype=np.float32):
        super().__init__()
        self.in_size = in_size # we can leave this None, and get from data in forward
        self.out_size = out_size
        self.dtype = dtype

        # init W. if in_size not specified, init later (when forward called)
        self.W = Parameter(None, name='W')
        if self.in_size is not None:
            self._init_W()
        # init bias
        self.b = None if nobias else Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        # http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()
        
        y = F.linear(x, self.W, self.b)
        return y