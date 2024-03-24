import numpy as np
import scipy.stats as st
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

f = lambda x: (x ** 2)
x_tr = np.linspace(0., 3, 200)
y_tr = f(x_tr)
plt.figure(figsize=(6, 6))
axes = plt.gca()
axes.set_xlim([0, 3])
axes.set_ylim([0, 8])
plt.plot(x_tr[:200], y_tr[:200], '--k')
plt.show()


x = np.array([.2, .5, .8, .9, 1.3, 1.7, 2.1, 2.7])
y = f(x) + np.random.randn(len(x))
plt.figure(figsize=(6, 6))
axes = plt.gca()
axes.set_xlim([0, 3])
axes.set_ylim([0, 8])
plt.plot(x_tr, y_tr, '--k')
plt.plot(x, y, 'ok', ms=10)
plt.show()


# We create the model.
lr = lm.LinearRegression()
# We train the model on our training dataset.
lr.fit(x[:, np.newaxis], y)
print(lr.coef_)
print(lr.intercept_)

f_lr = lambda x: lr.coef_ * x +lr.intercept_

x_f_lr = np.linspace(0., 3, 200)
y_f_lr = f_lr(x_tr)
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_tr, y_tr, '--k')
plt.plot(x_f_lr, y_f_lr, 'g')
plt.plot(x, y, 'ok', ms=10)
plt.show()

point = np.array([1.5])
y_point = lr.predict(point[:, np.newaxis])

plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_tr, y_tr, '--k')
plt.plot(x_f_lr, y_f_lr, 'g')
plt.plot(x, y, 'ok', ms=10)
plt.plot(point, y_point, 'or', ms=10)
plt.show()

from sklearn.metrics import r2_score
print(lr.score(x[:, np.newaxis], y))
print(r2_score(lr.predict(x[:, np.newaxis]), y))

deg = 1
xx = np.vander(x, deg+1)

res = np.linalg.solve(np.dot(xx.transpose(), xx), np.dot(xx.transpose(), y.reshape(-1, 1)))
print(res)

plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x, y, '--k', label='points')
plt.plot(x, f(x), 'g', label='square function')
eq = lambda x: res[0] * x + res[1]
plt.plot(x, eq(x), 'ok', ms=10, label='Equation solutions')
plt.legend()
plt.show()