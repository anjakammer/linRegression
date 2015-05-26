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
xLists = np.array([np.random.uniform(x_min, x_max, features_with_X0) for x in range(m)]) # 3x 100 mit random ziffern
xLists[:, 0] = 1.0  # erste spalte durch 1.0 ersetzen



# 2
# Theatas => Ebenengleichung
thetas = np.array([1.1, 2.0, -.9]) # diese originalen Thetas sollen am Ende wieder gefunden werden => Trainingsdaten


def linear_hypothesis(thetas):
    return lambda xLists: xLists.dot(np.transpose(thetas))

h = linear_hypothesis(thetas)


# 3 a
#y = h(xLists) + 2. * np.random.randn(m)
y = h(xLists)       # Vector mit 100 werten
print("Y-values: \n" + str(y))
#print("geshaptes y : %s" % y.shape)


y_new = y + np.random.randn(m) * 2          # *2 um Rauschen zu verstaerken
print("Y-values with gauss rauschen: \n" + str(y_new))

#3 b
x1 = xLists[:, 1]  # 1 vector
print("x1-values: \n" + str(x1))

x2 = xLists[:, 2]  # 2 vector
print("x2-values: \n" + str(x2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 1x1 grid, 1 subplot
ax.scatter(x1, x2, y_new)
ax.set_xlabel("feature x1")
ax.set_ylabel("feature x2")
ax.set_zlabel("Y-Values")
plt.show()

#4 cost function
def cost_function(x, y):
    assert (len(x) == len(y))
    m = len(x)
    return lambda thetas: (1. / (2. * float(m)) * (linear_hypothesis(thetas)(x) - y) ** 2).sum()


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
    costs = np.zeros(nb_iterations) # vector von nullen
    for i in range(nb_iterations):
        theta = compute_new_theta(x, y, theta, alpha)
        costs[i] = cost_f(theta)
        #print(cost_f(theta))
    return theta, costs

alpha = 0.001
iter = 1000     # iterationen => stop condition
start_thetas = np.array([1.5, 1.0, 0.0])
new_thetas, costs = gradient_decent(alpha, start_thetas, iter, xLists, y_new)
print("gelernte thetas: \n" + str(new_thetas)) # neue Thetas
print("originale thetas: \n" + str(thetas)) # originale Thetas

# 5c
plt.plot(np.array(range(iter)), costs)
plt.show()

# 6
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot(xLists[:, 1], xLists[:, 2], linear_hypothesis(new_thetas)(xLists))

# Code von Simon zur Darstellung der Hyperebene
plainFunc = lambda x, y: new_thetas[0] + new_thetas[1] *x + new_thetas[2]*y # bauen der Ebenengleichung
# create meshgrid to plot a surface
plainX, plainY = np.meshgrid(np.linspace(-10., 10.), np.linspace(-10., 10.))    # range in der die Ebene sich ausbreitet
ax.plot_surface(plainX, plainY, plainFunc(plainX, plainY))  # Ebene plotten
ax.scatter(x1, x2, y_new)

ax.scatter(xLists[:, 1], xLists[:, 2], y_new)
ax.set_xlabel("feature x1")
ax.set_ylabel("feature x2")
ax.set_zlabel("Y-Values")
plt.show()






