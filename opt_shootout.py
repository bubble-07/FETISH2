import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import secant
def rosenbrock(v):
    return ((1.0 - v[0]) ** 2) + 100.0 * ((v[1] - (v[0] ** 2)) ** 2)

start = 900
end = 1000

X_lin = np.array([[2.2, 2.3], [2.4, 2.6]])
Y_lin = np.array([[646.6], [1000.52]])
target = np.array([0])
for i in range(end):
    X_lin, Y_lin = secant.memory_secant(rosenbrock, (X_lin, Y_lin), target)
print X_lin
xs = X_lin[:, 0]
ys = X_lin[:, 1]
xs = xs[start:]
ys = ys[start:]
plt.scatter(xs, ys)

plt.show()
