import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.linear_model as lm
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

f = lambda x: ((x*2-1)*(x**2-2)*(x-2)+3)
x_tr = np.linspace(0, 3, 200)
y_tr = f(x_tr)
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_tr[:200], y_tr[:200], '--k');
plt.show()

x = stats.uniform(0,3).rvs(100)
y = f(x) + stats.norm(0,0.2).rvs(len(x))
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.plot(x_tr, y_tr, '--k');
plt.plot(x, y, 'ok', ms=10);
plt.show()

x=np.vstack(x)
model1 = linear_model.LinearRegression()
model1.fit(x, y)

print(model1.coef_)
print(model1.intercept_)
print(model1.score(x,y))

# Plot outputs
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.scatter(x, y,  color='black')
plt.plot(x, model1.predict(x), color='blue',linewidth=3)
plt.show()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model2 = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
model2.fit(x, y)

# Plot outputs
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.scatter(x, y,  color='black')
x_plot = np.vstack(np.linspace(0, 10, 100))
plt.plot(x_plot, model2.predict(x_plot), color='blue',linewidth=3)
plt.show()

model3 = make_pipeline(PolynomialFeatures(3), linear_model.LinearRegression())
model3.fit(x,y)

plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.scatter(x, y,  color='black')
x_plot = np.vstack(np.linspace(0, 10, 100))
plt.plot(x_plot, model3.predict(x_plot), color='blue',linewidth=3)
plt.show()

model4 = make_pipeline(PolynomialFeatures(4), linear_model.LinearRegression())
model4.fit(x,y)

plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.scatter(x, y,  color='black')
x_plot = np.vstack(np.linspace(0, 10, 100))
plt.plot(x_plot, model4.predict(x_plot), color='blue',linewidth=3)
plt.show()

model5 = make_pipeline(PolynomialFeatures(5), linear_model.LinearRegression())
model5.fit(x,y)

plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.scatter(x, y,  color='black')
x_plot = np.vstack(np.linspace(0, 10, 100))
plt.plot(x_plot, model5.predict(x_plot), color='blue',linewidth=3)
plt.show()



plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,3])
axes.set_ylim([0,8])
plt.scatter(x, y,  color='black')
x_plot = np.vstack(np.linspace(0, 10, 100))
plt.plot(x_plot, model1.predict(x_plot), color='red',linewidth=3, label='1')
plt.plot(x_plot, model2.predict(x_plot), color='green',linewidth=3, label='2')
plt.plot(x_plot, model3.predict(x_plot), color='yellow',linewidth=3, label='3')
plt.plot(x_plot, model4.predict(x_plot), color='brown',linewidth=3, label='4')
plt.plot(x_plot, model5.predict(x_plot), color='blue',linewidth=3, label='5')
plt.legend()
plt.show()

from sklearn import  metrics

models = [model1, model2, model3, model4, model5]
for model in models:
    print(model)
    error1 = np.average( np.abs(model.predict(x) -y) )
    print("Mean absolute errors: {}".format(error1))
    print("Mean absolute errors: {}".format(metrics.mean_absolute_error(y, model1.predict(x))))

    error2 = np.average( (model.predict(x) -y) **2 )
    print("Mean squared error: {}".format(error2))
    print("Mean squared error: {}".format( metrics.mean_squared_error(y, model1.predict(x)) ))

    error3 = np.median( np.abs(model.predict(x) -y) )
    print("Median absolute error: {}".format( error3 ))
    print("Median absolute error: {}".format( metrics.median_absolute_error(y, model1.predict(x)) ))

    print("R^2: {}".format(metrics.r2_score(y, model.predict(x))))
    ss_res=np.sum( (y-model1.predict(x))**2 )
    ss_tot=np.sum( (y-np.mean(y))**2 )
    R=1-ss_res/ss_tot
    print("R^2: {}".format(R))

    error4 = 1-np.var(y - model.predict(x) )/np.var(y)
    print("Explained variance score: {}".format( error4 ))
    print("Explained variance score: {}".format( metrics.explained_variance_score(y, model1.predict(x)) ))
    print()
    print()
