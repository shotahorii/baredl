"""
https://ruder.io/optimizing-gradient-descent/index.html
"""

from abc import ABCMeta, abstractmethod
from .core import get_array_module
from .layers import Layer

class Optimiser(metaclass=ABCMeta):
    """
    Base class for optimisers.

    Parameters
    ----------
    params: Layer (or Model), array-like, or generator
        List of parameters to be optimised via the optimiser.
    """
    def __init__(self, params):
        if isinstance(params, Layer):
            params = params.parameters()
        self.params = [p for p in params]
        self.hooks = []

    def step(self):
        # list parameters containing not None grad
        params_to_update = [p for p in self.params if p.grad is not None]

        # preprocess (optional)
        for f in self.hooks:
            f(params_to_update)

        for p in params_to_update:
            self.update_one(p)

    def zero_grad(self):
        for param in self.params:
            param.cleargrad()

    @abstractmethod
    def update_one(param):
        pass

    def add_hook(self, f):
        self.hooks.append(f)


"""
class SGD(Optimiser):
    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data
"""


class SGD(Optimiser):
    """
    SGD with Momentum and Nesterov accelerated gradient.
    momentum = 0 and  nesterov = False by default, 
    which is vanilla SGD.
    """
    def __init__(self, params, lr=0.01, momentum=0.0, nesterov=False):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.vs = {}

    def update_one(self, param):

        # initialise v for the param, if this is first iter.
        v_key = id(param)
        if v_key not in self.vs:
            xp = get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        if self.nesterov:
            v += self.lr * param.grad.data * (param.data - v)
        else:
            v += self.lr * param.grad.data
        param.data -= v


class Adagrad(Optimiser):
    def __init__(self, params, lr=0.01, eps=1e-8):
        super().__init__(params)
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param):
        xp = get_array_module(param.data)

        # initialise h for the param, if this is first iter.
        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = xp.zeros_like(param.data)

        grad = param.grad.data
        h = self.hs[h_key]

        h += grad * grad
        param.data -= self.lr * grad / xp.sqrt(h + self.eps)