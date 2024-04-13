import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.stats as stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import model_selection


true_fun = lambda X: np.cos(1.5 * np.pi * X)
n_samples=20
x = np.sort(np.random.rand(n_samples))
y = true_fun(x) + np.random.randn(n_samples) * 0.1
x=np.vstack(x)


clf = Ridge(alpha=1.0)
clf.fit(x, y)

x_plot = np.vstack(np.linspace(0, 1, 20))
plt.plot(x_plot, clf.predict(x_plot), color='blue',linewidth=3)
plt.plot(x, y, 'ok');
plt.show()



# prepare models
models = []
predicts = []
names=[]
models.append(('LR degree 2', make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression()) ))
models.append(('LR degree 20', make_pipeline(PolynomialFeatures(20), linear_model.LinearRegression()) ))

x_plot = np.vstack(np.linspace(-3, 3, 1000))
for name, model in models:
    print(name)
    model.fit(x, y)
    predicts.append(model.predict(x_plot))
    names.append(name)
print()
x_plot = np.vstack(np.linspace(-3, 3, 1000))
plt.plot(x, y, 'ok');
for i in range(len(models)):
    #print(i)
    plt.plot(x_plot, predicts[i],linewidth=3,label=names[i])
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
plt.legend()
plt.show()

# Zadanie 1
# =====================================================

models = []
predicts = []
names=[]
models.append(('LR 20', make_pipeline(PolynomialFeatures(20), linear_model.Ridge()) ))
models.append(('RR 20 1', make_pipeline(PolynomialFeatures(20), linear_model.Ridge(alpha=1))))
models.append(('RR 20 10000', make_pipeline(PolynomialFeatures(20), linear_model.Ridge(alpha=10000))))
models.append(('RR 20 0.0001', make_pipeline(PolynomialFeatures(20), linear_model.Ridge(alpha=0.0001))))


for name, model in models:
    print(name)
    model.fit(x, y)
    predicts.append(model.predict(x_plot))
    names.append(name)
print()
x_plot = np.vstack(np.linspace(-3, 3, 1000))
plt.plot(x, y, 'ok');
for i in range(len(models)):
    #print(i)
    plt.plot(x_plot, predicts[i],linewidth=3,label=names[i])
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
plt.legend()
plt.show()

# Zadanie 2
# ===================================================

seed=123
kfold = model_selection.KFold(n_splits=10)
scoring = 'neg_mean_absolute_error'
#scoring = 'r2'


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), linear_model.Ridge(alpha=1)),
                    param_grid={'polynomialfeatures__degree': [1, 2, 3, 4, 5, 6, 7],
                                'ridge__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100 ]},
                    cv=kfold,
                    refit=False)
#make_pipeline(PolynomialFeatures(degree=2), linear_model.LinearRegression()).get_params().keys()
grid.fit(x, y)
print(grid.best_params_)

# Zadanie 3
# ================================================

import pandas as pd
df_adv = pd.read_csv('https://raw.githubusercontent.com/przem85/bootcamp/master/statistics/Advertising.csv', index_col=0)
X = df_adv[['TV', 'radio','newspaper']]
y = df_adv['sales']
df_adv.head()


grid = GridSearchCV(make_pipeline(PolynomialFeatures(degree=2), linear_model.Ridge(alpha=1)),
                    param_grid={'polynomialfeatures__degree': [1, 2, 3, 4, 5, 6, 7],
                                'ridge__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100 ]},
                    cv=kfold,
                    refit=False)
#make_pipeline(PolynomialFeatures(degree=2), linear_model.LinearRegression()).get_params().keys()
grid.fit(X, y)
print(grid.best_params_)
from sklearn.metrics import r2_score

model = make_pipeline(PolynomialFeatures(
    grid.best_params_['polynomialfeatures__degree']),
    Ridge(alpha=grid.best_params_['ridge__alpha']))
model.fit(X,y)
print(r2_score(y, model.predict(X)))
