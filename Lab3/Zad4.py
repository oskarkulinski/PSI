import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

f = lambda x: x**2
x_tr = np.linspace(-10, 10, 200)
y_tr = f(x_tr)
x = stats.uniform(-3,6).rvs(100)
y = f(x) + stats.norm(0,0.3).rvs(len(x))

plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([-3,3])
axes.set_ylim([0,10])
# axes.set_ylim([np.min(y),f(5)])
plt.plot(x_tr, y_tr, '--k');
plt.plot(x, y, 'ok', ms=3);

plt.show()
corr = {}
corr['pearson'], _ = stats.pearsonr(x,y)
corr['spearman'], _ = stats.spearmanr(x,y)
corr['kendall'], _ = stats.kendalltau(x,y)
print(corr)