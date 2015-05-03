__author__ = 'anja'
import numpy as np
from matplotlib import pylab as plt


# Uebung 1 Aufgabe 1)

# 1 x-werte -10 x>10
x_min = -10.
x_max = 10.
m = 11
x = np.random.uniform(x_min, x_max, m)
print(x)
a = 8
b = 5
y_without_noise = a * x + b
print(y_without_noise)
plt.plot(x, y_without_noise,'b*')
plt.show()                                           # plotten
y = y_without_noise + 9. * np.random.randn(11)       # zieht uns ein Sample aus der Gaussverteilung
# np.random.randn(11)                                # Array mit 11 gaussverteilten Werten
plt.plot(x, y, 'b*')
plt.show()
for i in zip(x,y):           #nimmt 2 Listen und macht ein Paerchen -> zip iteriert ueber 2 Listen
    print(i)

# Uebung 1 Aufgabe 2)
def linear_hypothesis(theta_0, theta_1):
    return lambda x:theta_0 + theta_1 * x           # Geradengleichung
h = linear_hypothesis(theta_0=1, theta_1=2)
print(h(1))                                         # => 3 bester Y-Wert fuer einen x-Wert
print(h(np.array([1., 2.])))                        # array([3., 5.]) -> 2 y-Werte fuer 2 x-Werte

# Uebung 1 Aufgabe 3)
def cost_function(x, y):
    assert type(x) == np.ndarray
    assert type(y) == np.ndarray
    assert len(x) == len(y)
    m = len(x)
    return lambda theta_0, theta_1: 1. / (2. * m) * \
            ((linear_hypothesis(theta_0, theta_1)(x)-y)**2).sum()       # funktionsaufruf von Variable function()()

cost = cost_function(x, y)                          # 2 Arrays mit x und y Werten
print(cost(5., 9.))                                 # 2 thetas als Parameter (Geradengleichung)-> Ergebnis ist kosten der gerade (optimum)


# Uebung 1 Aufgabe 4)
ran = 20.
t0 = np.arange(a - ran, a + ran, ran * 0.05)
t1 = np.arange(b - ran, b + ran, ran * 0.05)

C = np.zeros([len(t0),len(t1)])                 # Eine Matrix aus Nullen in den Dimensionen der Anzahl von Theta Werten
squared_error_cost = cost_function
c =  squared_error_cost(x, y)

for i, t_0 in enumerate(t0):
  for j, t_1 in enumerate(t1):
    C[j][i] = c(t_0, t_1)

T0, T1 = np.meshgrid(t0, t1)

plt.subplot(121)
plt.contour(T0, T1, C)
plt.xlabel('$\Theta_0$')
plt.ylabel('$\Theta_1$')
plt.title('Kostenfunktion')
plt.show()

# Ueung 1 Aufgabe 5) Gradientenabstiegsverfahren
# Gradient -> Vektor der in die steilste Richtung zeigt -> Gegenteil unseres Optimums
def compute_new_theata(x, y, theta_0, theta_1, alpha):
    assert type(x) == np.ndarray
    assert type(y) == np.ndarray
    assert len(x) == len(y)
    m = len(x)
    #update Rule
    temp_0 = theta_0 - alpha * 1. / m * (theta_0 + theta_1 * x - y).sum()
    temp_1 = theta_0 - alpha * 1. / m * ((theta_0 + theta_1 * x - y) * x).sum()
    theta_0 = temp_0
    theta_1 = temp_1
    return temp_0, temp_1



theta_0 = 1.                # Startwerte -> Selbst festgelegt
theta_1 = 2.
alpha = 0.005                # Schrittrate/Schrittweite
costs = {}

for i in range(20000):
    theta_0, theta_1 = compute_new_theata(x, y, theta_0, theta_1, alpha)
    costs[i] = cost(theta_0, theta_1)

print(theta_1, theta_0)     # nach 2000 Iterationen

plt.plot(x, y, 'b*')        # Blaue Sterne als unsere gausswerte
x_ = np.array([-10., 10.])
h_ = theta_0 + theta_1 * x_
plt.plot(x_, h_, 'r-')      # Die optimalste Gerade nach 20000 Iterationen
plt.show()

plt.plot(costs.keys(), costs.values(), 'r-')
plt.show()

# als Hausaufgabe ein Graph X-Achse = Iterationen, Y-Achse = Kosten
# gehen kosten hoch/runter im Wechsel, ist alpha zu gross
# kosten muessen konstant nach unten gehen, je schneller - desto besser das alpha











