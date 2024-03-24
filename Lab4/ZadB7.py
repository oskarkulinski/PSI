import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt
from ZadB3 import compute_error
from ZadB5 import compute_error_1

f = lambda x: (x)
x_tr = np.linspace(0., 3, 200)
y_tr = f(x_tr)
x = np.array([.2, .5, .8, .9, 1.3, 1.7, 2.1, 2.7])
y = f(x) + np.random.randn(len(x))/5
y[1]=y[1]+10
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,12])
plt.plot(x_tr, y_tr, '--k')
plt.plot(x, y, 'ok', ms=10)
plt.show()

points = np.column_stack((x, y))

sqr_err = opt.fmin_cg(compute_error, np.array((0, 0)),
                  args=(x.transpose(), y.transpose()))

abs_err = opt.fmin_cg(compute_error_1, np.array((0, 0)),
                      args=(x.transpose(), y.T))

plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,12])
plt.plot(x, y, '--k', label='points')
plt.plot(x, f(x), 'g', label='Function')
regression = lambda x,err: err[0] * x + err[1]
plt.plot(x, regression(x, sqr_err) , ms=10, label='Squared Error')
plt.plot(x, regression(x,abs_err), ms=10, label='Absolut error')
plt.legend()
plt.show()
