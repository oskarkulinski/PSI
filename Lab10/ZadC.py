import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.plot(X, y, "b.")
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.axis([0, 2, 0, 15])
plt.show()

X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print(theta_best)

class MyRegression:
    def __init__(self, theta):
        self.coefs = (theta[0], theta[1])

    def predict(self, X):
        return X.dot(self.coefs)

X_new = X_b
reg = MyRegression(theta_best)
y_predict = reg.predict(X_new)

plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X_new, y)
y_predict_lin = lin_reg.predict(X_new)

plt.plot(X_new, y_predict_lin, "r-")
plt.plot(X, y, "b.")
plt.show()




eta = 0.1  # learning rate
n_iterations = 1000
m = 100
theta_path_bgd = []

theta_0 = np.random.randn(2,1)  # random initialization

theta = theta_0

for i in range(n_iterations):
    gradient = 2/m * X.T.dot(theta.T.dot(X) - y)
    theta = theta - eta * gradient
    theta_path_bgd.append(theta)

print(theta_path_bgd)

# SGD #########################################

theta_path_sgd = []
m = len(X_b)
np.random.seed(42)

eta = 0.1  # learning rate
n_iterations = 1000
m = 100

theta = theta_0

for i in range(n_iterations):
    for i in range(X):
        gradient = 2/m * X[i].dot(theta.T.dot(X[i]) - y[i])
        theta = theta - eta * gradient
        theta_path_sgd.append(theta)


# MGD ######################################
theta_path_mgd = []

n_iterations = 50
minibatch_size = 20

theta = theta_0

import random

for i in range(n_iterations):
    batch_indexes = random.sample(range(X), minibatch_size)
    X_batch = X[batch_indexes]
    y_batch = y[batch_indexes]
    gradient = 2/m * X_batch.dot(theta.T.dot(X_batch) - y_batch)
    theta = theta - eta * gradient
    theta_path_sgd.append(theta)


theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

plt.plot(range(n_iterations), theta_path_bgd, "b.")
plt.plot(range(n_iterations), theta_path_sgd, "r.")
plt.plot(range(n_iterations), theta_path_mgd, "b.")