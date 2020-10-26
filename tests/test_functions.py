import sys
sys.path.append(sys.path[0][:-5])

import math
import numpy as np
from baredl import Tensor
from baredl.functions import exp
from baredl.utils import numerical_diff


def test_exp():
    var = np.random.rand(1) # var is a np.array([x])
    print('testtt')
    print(var)

    x = Tensor(np.array(var))
    y = exp(x)
    expected = math.exp(var)
    assert np.allclose(y.data, expected)


def test_exp_trad():
    var = np.random.rand(1) # var is a np.array([x])

    x = Tensor(np.array(var))
    y = exp(x)
    y.backward()
    num_grad = numerical_diff(exp, x)
    assert np.allclose(x.grad.data, num_grad)
