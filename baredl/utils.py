import numpy as np


def logsumexp(x, axis=1):
    """
    https://blog.feedly.com/tricks-of-the-trade-logsumexp/
    """
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m