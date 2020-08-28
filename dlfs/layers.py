import weakref
import numpy as np
from dlfs.core import Parameter, Layer
import dlfs.functions as F

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