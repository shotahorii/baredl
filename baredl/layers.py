import os
import weakref
from abc import ABCMeta, abstractmethod
import numpy as np
from baredl.core import Parameter, get_array_module
import baredl.functions as F


class Layer(metaclass=ABCMeta):
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
    
    @abstractmethod
    def forward(self, inputs):
        pass

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

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict, parent_key=''):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu() # always save as np.ndarray not cp.ndarray

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            # if the save operation is interrupted by user input (such as Ctrl+C)
            # then remove the work-in-progress file
            if os.path.exists(path):
                os.remove(path) 
            raise

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


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

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        # http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        W_data = xp.random.randn(I, O).astype(self.dtype) * xp.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = get_array_module(x)
            self._init_W(xp)
        
        y = F.linear(x, self.W, self.b)
        return y