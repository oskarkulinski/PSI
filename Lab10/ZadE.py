import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.linear_model as lm
from sklearn import datasets
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

f = lambda x: ((x * 2 - 1) * (x ** 2 - 2) * (x - 2) + 3)
x_tr = np.linspace(0, 3, 200)
y_tr = f(x_tr)
x = stats.uniform(0, 3).rvs(100)
y = f(x) + stats.norm(0, 0.2).rvs(len(x))

x = np.vstack(x)
x_plot = np.vstack(np.linspace(0, 10, 100))

MLP = MLPRegressor(hidden_layer_sizes=(100, 50, 10), activation='tanh', max_iter=50000, batch_size=20,
                   learning_rate_init=0.001, learning_rate="adaptive", solver='adam')
y_rbf = MLP.fit(x, y)

# Plot outputs
plt.figure(figsize=(6, 6));
axes = plt.gca()
axes.set_xlim([0, 3])
axes.set_ylim([0, 8])
plt.scatter(x, y, color='black')
plt.plot(x_plot, MLP.predict(x_plot), color='blue', linewidth=3)
plt.show()

from sklearn import metrics

print(metrics.r2_score(y, MLP.predict(x)))

from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
scores = []
for train_index, test_index in kf.split(x_tr, y_tr):
    X_train, X_test = x_tr[train_index], x_tr[test_index]
    y_train, y_test = y_tr[train_index], y_tr[test_index]

    MLP = MLPRegressor(hidden_layer_sizes=(100, 50, 10), activation='tanh',
                       max_iter=50000, batch_size=20, learning_rate_init=0.001,
                       learning_rate="adaptive", solver='adam')
    MLP.fit(X_train.reshape(-1, 1), y_train)

    y_pred = MLP.predict(X_test.reshape(-1, 1))

    r2 = metrics.r2_score(y_test, y_pred)

    scores.append(r2)

print(np.mean(scores))

#################################################

import pandas as pd

df_adv = pd.read_csv('Advertising.xls', index_col=0)
print(df_adv)
X = df_adv[['TV', 'Radio', 'Newspaper']]
y = df_adv['Sales']

grid = GridSearchCV(MLP, cv='kfold', param_grid={'hidden_layer_sizes': [100, 50, 10]})
grid.fit(X, y)

print(metrics.r2_score(y_true=y, y_pred=grid.best_estimator_.predict(X)))
