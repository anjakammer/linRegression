# 1 
from matplotlib import pylab as plt
import numpy as np
x_min = -10.
x_max = 10.
m = 11
x = np.array([np.random.uniform(x_min, x_max, m),np.random.uniform(x_min, x_max, m)])
print(x)

# 2

def linear_hypothesis(theta_0, theta_1):
    return lambda x:theta_0 + theta_1 * x #Ebenengleichung

h = linear_hypothesis(theta_0=1, theta_1=2)
print(h(1)) # => 3 bester Y-Wert fuer einen x-Wert
print(h(np.array([1., 2.])))
