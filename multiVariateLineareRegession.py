# 1 
from matplotlib import pylab as plt
import numpy as np
x_min = -10.
x_max = 10.
m = 2
xList = np.random.uniform(x_min, x_max, m)
xList[0] = 1 
print(xList)

# 2
thetas = [2,3]
def linear_hypothesis(thetas):
	return lambda xList: np.sum([ theta * x for theta, x in zip(thetas, xList)])

h = linear_hypothesis(thetas)
print(h(x))
