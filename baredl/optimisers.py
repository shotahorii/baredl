from abc import ABCMeta, abstractmethod
from bareml.core import get_array_module


class Optimiser(metaclass=ABCMeta):
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def step(self):
        # list parameters containing not None grad
        params = [p for p in self.target.params() if p.grad is not None]

        # preprocess (optional)
        for f in self.hooks:
            f(params)

        for p in params:
            self.update_one(p)

    def zero_grad(self):
        self.target.zero_grad()

    @abstractmethod
    def update_one(param):
        pass

    def add_hook(self, f):
        self.hooks.append(f)


class SGD(Optimiser):
    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimiser):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()
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
