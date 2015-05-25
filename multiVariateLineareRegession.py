# 1
from matplotlib import pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x_min = -10.
x_max = 10.
m = 100
features_count = 2
features_with_X0 = features_count + 1
# np.concatenate
xLists = np.array([np.random.uniform(x_min, x_max, features_with_X0) for x in range(m)])
xLists[:, 0] = 1.0  # first column to 1.0



# 2
thetas = np.array([1.1, 2.0, -.9])


def linear_hypothesis(thetas):
    return lambda xLists: xLists.dot(np.transpose(thetas))


h = linear_hypothesis(thetas)
print(h(xLists))


# 3 a
#y = h(xLists) + 2. * np.random.randn(m)
y = h(xLists)
print("Y-values: \n" + str(y))

y_noise_simga = y * 0.1
y_new = y + np.random.randn(m) * y_noise_simga
print(y_new)
print("Y-values with gauss rauschen: \n" + str(y_new))

#3 b
x1 = xLists[:, 1]  # 1 vector
print("x1-values: \n" + str(x1))

x2 = xLists[:, 2]  # 2 vector
print("x2-values: \n" + str(x2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 1x1 grid, 1 subplot
ax.scatter(x1, x2, y_new)
plt.show()

#4 cost function
def cost_function(x, y):
    assert (len(x) == len(y))
    m = len(x)
    return lambda thetas: 1. / ((2. * float(m)) * (linear_hypothesis(thetas)(x) - y) ** 2).sum()


j = cost_function(xLists, y_new)
print("cost-function: \n" + str(j(thetas)))

# 5a
def compute_new_theta(x, y, thetas, alpha):
    m = y.size
    return thetas - alpha * (1. / m) * \
                    x.transpose().dot(linear_hypothesis(thetas)(x) - y)


# 5b
def gradient_decent(alpha, theta, nb_iterations, x, y):
    cost_f = cost_function(x, y)
    costs = np.zeros(nb_iterations)
    for i in range(nb_iterations):
        new_thetas = compute_new_theta(x, y, theta, alpha)
        costs[i] = cost_f(new_thetas)
    return new_thetas, costs

alpha = 0.01
iter = 1000     # iterationen
new_thetas, costs = gradient_decent(alpha, thetas, iter, xLists, y_new)
print new_thetas # neue Thetas

# 5c
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.array(range(iter)), costs)
plt.show()

# 6
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xLists[:, 1], xLists[:, 2], linear_hypothesis(new_thetas)(xLists))
ax.scatter(xLists[:, 1], xLists[:, 2], y_new)
plt.show()






