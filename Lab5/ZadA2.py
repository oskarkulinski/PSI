import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
from ZadA1 import f_lr_1, f_lr_2

f = lambda x: ((x*2-1)*(x**2-2)*(x-2)+3)

x_tr = np.linspace(0, 3, 200)
y_tr = f(x_tr)
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_tr[:200], y_tr[:200], '--k')
plt.show()


x = stats.uniform(0,3).rvs(100)
y = f(x) + stats.norm(0,0.2).rvs(len(x))
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_tr, y_tr, '--k')
plt.plot(x, y, 'ok', ms=10)
plt.show()

M1 = np.vstack( (np.ones_like(x), x, x**2, x**3) ).T
p1 = np.linalg.lstsq(M1, y, rcond=None)

f_lr_3 = lambda x: p1[0][1] * x +p1[0][2] * x**2 + p1[0][3] * x**3 + p1[0][0]

x_f_lr = np.linspace(0., 3, 200)
y_f_lr = f_lr_1(x_tr)
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_f_lr, y_f_lr, 'g');
plt.plot(x, y, 'ok', ms=10);
plt.show()


M2 = np.vstack( (np.ones_like(x), x, x**2, x**3, x**4) ).T
p2 = np.linalg.lstsq(M2, y, rcond=None)

f_lr_4 = lambda x: (p2[0][1] * x +p2[0][2] * x**2 + p2[0][3] * x**3 +
                    p2[0][4] * x**4 + p2[0][0])

x_f_lr = np.linspace(0., 3, 200)
y_f_lr = f_lr_2(x_tr)
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_f_lr, y_f_lr, 'g');
plt.plot(x, y, 'ok', ms=10);
plt.show()

M3 = np.vstack( (np.ones_like(x), x, x**2, x**3, x**4, x**5) ).T
p3 = np.linalg.lstsq(M3, y, rcond=None)

f_lr_5 = lambda x: (p3[0][1] * x +p3[0][2] * x**2 + p3[0][3] * x**3 +
                    p3[0][4] * x**4 + p3[0][5] * x**5 + p3[0][0])

x_f_lr = np.linspace(0., 3, 200)
y_f_lr = f_lr_3(x_tr)
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_f_lr, y_f_lr, 'g');
plt.plot(x, y, 'ok', ms=10);
plt.show()


x_f_lr = np.linspace(0., 3, 200)
y_f_lr_1 = f_lr_1(x_tr)
y_f_lr_2 = f_lr_2(x_tr)
y_f_lr_3 = f_lr_3(x_tr)
y_f_lr_4 = f_lr_4(x_tr)
y_f_lr_5 = f_lr_5(x_tr)
plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_f_lr, y_f_lr_3, 'c', label="1 Regression")
plt.plot(x_f_lr, y_f_lr_3, 'y', label="2 Regression")
plt.plot(x_f_lr, y_f_lr_3, 'r', label="3 Regression")
plt.plot(x_f_lr, y_f_lr_4, 'g', label="4 Regression")
plt.plot(x_f_lr, y_f_lr_5, 'b', label='5 Regression')
plt.plot(x, y, 'ok', ms=10)
plt.legend()
plt.show()
