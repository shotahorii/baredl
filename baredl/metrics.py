import numpy as np
from baredl.core import Variable, as_variable, as_array

def accuracy(y, t):
    """
    y: baredl.Variable or np.ndarray (n, c)
        n: number of samples
        c: number of classes
        Assuming it contains probabilities for each class
        e.g. [[0.1,0.3,0.6], [0.1,0.8,0.1], ...]

    t: baredl.Variable or np.array (n,)
        n: number of samples
        Assuming it contains the true class label as index
        e.g. [2,1,1,0,2,0,...]
    """
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))