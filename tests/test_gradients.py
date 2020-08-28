import unittest
import numpy as np

import sys
sys.path.append(sys.path[0][:-5])

from dlfs import Variable
from dlfs.functions import numerical_diff

##############################
# Benchmark functions
##############################

def sphere(x, y):
    z = x ** 2 + y ** 2
    return z

def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

##############################
# Test cases
##############################

class GradientTest(unittest.TestCase):

    def test_sphere(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = sphere(x, y)
        z.backward()
        self.assertEqual(x.grad, 2)
        self.assertEqual(y.grad, 2)

    def test_matyas(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = matyas(x, y)
        z.backward()
        self.assertAlmostEqual(x.grad, 0.04)
        self.assertAlmostEqual(y.grad, 0.04)

    def test_goldstein(self):
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = goldstein(x, y)
        z.backward()
        self.assertEqual(x.grad, -5376)
        self.assertEqual(y.grad, 8064)



unittest.main() 
