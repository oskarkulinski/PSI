import numpy as np
import scipy.optimize as opt


def compute_error(ab, x, y):
    x = np.array(x).transpose()
    y = np.array(y).transpose()
    a, b = ab
    err = 0
    for i in range(len(x)):
        err += (y[i] - (a * x[i] + b)) ** 2
    return err


print(compute_error((1, 1), [1, 2], [2, 2]))

# Zad4
f = lambda x: (x ** 2)
x = np.array([.2, .5, .8, .9, 1.3, 1.7, 2.1, 2.7])
y = f(x) + np.random.randn(len(x))

res = opt.fmin_cg(compute_error, np.array((0, 0)),
                  args=(x.transpose(), y.transpose()))
print(res)
