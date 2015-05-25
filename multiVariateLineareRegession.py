# 1 
from matplotlib import pylab as plt
import numpy as np
x_min = -10.
x_max = 10.
m = 2
xList = np.random.uniform(x_min, x_max, m)
xList[0] = 1 
print(xList)

xList.reshape([len(xList), 1]).shape
# im Scatter plot darstellen - z-label ist das y, X und Y Label sind die x-werte

# 2
thetas = [2,3]
def linear_hypothesis(thetas):
	return lambda xList: np.sum([ theta * x for theta, x in zip(thetas, xList)])

h = linear_hypothesis(thetas)
x = [1,2]
print(h(x))


