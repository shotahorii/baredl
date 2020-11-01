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


class SGD(Optimiser):
    def __init__(self, params, lr=0.01):
        super().__init__(params)
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimiser):
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v
