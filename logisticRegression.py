
from matplotlib import pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#1
def logistic_function(z):
    return float(1 / (1 + np.exp(-z)))

iterations = 1000
plt.axis([-5, 5, 0, 1])
output = np.empty(iterations)
input = np.empty(iterations)

for i in range(iterations):
    input[i] = float((i/100) - 5.0)
    output[i] = (logistic_function(input[i]))

plt.plot(input, output)
plt.show()

#2
thetas = np.array([1.1, 2.0, -.9])

def logistic_hypothesis(thetas):
    return lambda x: 1 / (1 + np.exp(-thetas.transpose()*x))

h = logistic_hypothesis(thetas)
x = np.array([-5, 0, 5])
print(h(x))

#3
def cross_entropy_loss(h, x, y):
    return lambda theta: 1/2 * (h(theta)(x)-y)**2