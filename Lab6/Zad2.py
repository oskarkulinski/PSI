import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn import  metrics
from sklearn import utils

np.random.seed(0)
n_samples = 30
true_fun = lambda X: np.cos(1.5 * np.pi * X)
X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1

# Plot outputs
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([-1.5,1.5])
plt.scatter(X, y,  color='black')
x_tr = np.linspace(0, 1, 200)
plt.show()

s=np.random.random_sample(n_samples)
s[s>0.5]=1
s[s<=0.5]=0

X1=X[s==1]
y1=y[s==1]
X2=X[s==0]
y2=y[s==0]

# Plot outputs
plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([-1.5,1.5])
plt.scatter(X1, y1,  color='blue')
plt.scatter(X2, y2,  color='red')
x_tr = np.linspace(0, 1, 200)
plt.show()

x1 = np.vstack(X1)
x2 = np.vstack(X2)
model1 = linear_model.LinearRegression()
model1.fit(x1, y1)
model2 = linear_model.LinearRegression()
model2.fit(x2, y2)

plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([-1.5,1.5])
plt.scatter(X1, y1,  color='blue')
plt.scatter(X2, y2, color='red')
x_plot = np.vstack(np.linspace(0, 10, 100))
plt.plot(x_plot, model1.predict(x_plot), color='blue',linewidth=3)
plt.plot(x_plot, model2.predict(x_plot), color='red',linewidth=3)
plt.show()

model3 = make_pipeline(PolynomialFeatures(20), linear_model.LinearRegression())
model3.fit(x1, y1)
model4 = make_pipeline(PolynomialFeatures(20), linear_model.LinearRegression())
model4.fit(x2, y2)

plt.figure(figsize=(6,6));
axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([-1.5,1.5])
plt.scatter(X1, y1,  color='blue')
plt.scatter(X2, y2, color='red')
x_plot = np.vstack(np.linspace(0, 10, 100))
plt.plot(X1, model3.predict(X1.reshape(-1, 1)), color='blue',linewidth=3)
plt.plot(X2, model4.predict(X2.reshape(-1, 1)), color='red',linewidth=3)
plt.show()



r1 = []
r2 = []
r3 = []
r4 = []
r5 = []
r6 = []
r7 = []
m1 = []
m2 = []
m3 = []
m4 = []
m5 = []
m6 = []
m7 = []


for i in range(100):
    s=np.random.random_sample(n_samples)
    s[s>0.5]=1
    s[s<=0.5]=0

    X1=X[s==1]
    y1=y[s==1]
    X2=X[s==0]
    y2=y[s==0]

    x1 = np.vstack(X1)
    x2 = np.vstack(X2)

    model1 = linear_model.LinearRegression()
    model2 = make_pipeline(PolynomialFeatures(1), linear_model.LinearRegression())
    model3 = make_pipeline(PolynomialFeatures(2), linear_model.LinearRegression())
    model4 = make_pipeline(PolynomialFeatures(3), linear_model.LinearRegression())
    model5 = make_pipeline(PolynomialFeatures(4), linear_model.LinearRegression())
    model6 = make_pipeline(PolynomialFeatures(5), linear_model.LinearRegression())
    model7 = make_pipeline(PolynomialFeatures(6), linear_model.LinearRegression())

    model1.fit(x1, y1)
    model2.fit(x1, y1)
    model3.fit(x1, y1)
    model4.fit(x1, y1)
    model5.fit(x1, y1)
    model6.fit(x1, y1)
    model7.fit(x1, y1)

    r1.append(metrics.r2_score(y2, model1.predict(x2)))
    r2.append(metrics.r2_score(y2, model2.predict(x2)))
    r3.append(metrics.r2_score(y2, model3.predict(x2)))
    r4.append(metrics.r2_score(y2, model4.predict(x2)))
    r5.append(metrics.r2_score(y2, model5.predict(x2)))
    r6.append(metrics.r2_score(y2, model6.predict(x2)))
    r7.append(metrics.r2_score(y2, model7.predict(x2)))


plt.figure(figsize=(6,6))
axes = plt.gca()
axes.set_xlim([0,100])
axes.set_ylim([0,1])
x_plot = np.vstack(np.linspace(0, 100, 100))

plt.plot(x_plot, np.mean(rm1), color='red',linewidth=3, label='1')
plt.plot(x_plot, np.mean(rm2), color='green',linewidth=3, label='2')
plt.plot(x_plot, np.mean(rm3), color='yellow',linewidth=3, label='3')
plt.plot(x_plot, np.mean(rm4), color='brown',linewidth=3, label='4')
plt.plot(x_plot, np.mean(rm5), color='blue',linewidth=3, label='5')
plt.plot(x_plot, np.mean(rm6), color='orange',linewidth=3, label='6')
plt.plot(x_plot, np.mean(rm7), color='purple',linewidth=3, label='7')
plt.legend()
plt.show()
