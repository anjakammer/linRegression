# 1 
from matplotlib import pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
x_min = -10.
x_max = 10.
m = 4
features_count = 2
features_with_X0 = features_count + 1 # add col x0 zu every row
xLists = np.array([np.random.uniform(x_min, x_max, features_with_X0) for x in range(m)])

xLists[:,0] =1.0 # set the 1. col to 1.0

#print(xLists)

# 2

thetas = np.array([1.1, 2.0, -.9])
def linear_hypothesis(thetas):
	return lambda xLists: xLists.dot(thetas)
#3 a
h = linear_hypothesis(thetas)
y = h(xLists)
#print("Y-values: ")
#print(y)
#print("Y-values with gauss rauschen:")
y_noise_simga = y * 0.1
y_new = y + np.random.randn(m) * y_noise_simga
#print(y_new)

#3 b
#print("X1: ")
x1 = xLists[:,1]

#print(x1)
#print("X2: ")
x2 = xLists[:,2]
#print(x2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y_new)

#4 cost function
def cost_function(x, y):
    assert(len(x) == len(y))
    m = len(x)
    return lambda thetas: 1. / ((2. * float(m)) * (linear_hypothesis(thetas)(x) - y) ** 2).sum()

j = cost_function(xLists,y_new)
print(j(thetas))









