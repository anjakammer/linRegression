
from matplotlib import pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def logistic_function(z):
    return 1 / (1 + np.exp(-z ))


range = np.array(range(10))
n = 42
print(logistic_function(n))
plt.plot(1., 10.)
plt.show()