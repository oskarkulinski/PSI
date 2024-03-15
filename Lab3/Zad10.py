import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import scipy.stats as stats
from scipy.stats import multivariate_normal

mean1 = np.array([0, 0])
cov1 = np.array([[4.40, -2.75], [-2.75,  5.50]])
X1_rv=multivariate_normal(mean1, cov1)
X = X1_rv.rvs(1000)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
ax.scatter(X[:, 0], X[:, 1])
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
plt.show()

means = X.mean(axis=0)
cov = np.cov(X.T)

X2_rv=multivariate_normal(means, cov)

X2 = X2_rv.rvs(1000)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
ax.scatter(X2[:, 0], X2[:, 1])
ax.contourf(X2)
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
plt.show()




eigenvalue, eigenvectors = np.linalg.eig(cov)
print(eigenvalue)
print(eigenvectors)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
ax.scatter(X2[:, 0], X2[:, 1])
ax.contourf(X2)
ax.scatter(eigenvectors[:, 0], eigenvectors[:, 1])
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
plt.show()



x, y = np.mgrid[0:1:.01, 0:1:.01]
square = np.dstack((x, y))
print(square.shape)
X3 = X2_rv.rvs(size=square.shape[0])

covs = np.cov(X3.T)
print(X3.mean(axis=0))
print(covs)

eigenvalue, eigenvectors = np.linalg.eig(covs)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
ax.scatter(X3[:, 0], X3[:, 1])
ax.contourf(X3)
ax.scatter(eigenvectors[:, 0], eigenvectors[:, 1])
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])
plt.show()



