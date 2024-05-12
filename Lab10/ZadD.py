import numpy as np
import sklearn.datasets


#low noise, plenty of samples, should be easy
X, y = sklearn.datasets.make_moons(n_samples=1000, noise=.05)


import matplotlib.pyplot as plt

plt.scatter(X[:,0],X[:,1],c=y)
plt.axis('equal')
plt.show()


from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression


from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())


X = iris["data"][:, 3:] # petal width
y = (iris["target"] == 2).astype(np.int32) # 1 if Iris-Virginica, else 0


reg = LogisticRegression()
reg.fit(X, y)

print(reg.score(X, y))
plot_decision_regions(X, y, reg)
plt.show()

###############################################################3


X = iris["data"][:, 0:2]
y = (iris["target"] == 2).astype(np.int32) # 1 if Iris-Virginica, else 0

plt.scatter(X[:, 0], X[:, 1], c=y)


reg2 = LogisticRegression()
reg2.fit(X, y)

print(reg2.score(X, y))
plot_decision_regions(X, y, reg2)
plt.show()

#######################################################

X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"]


plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

reg3 = LogisticRegression(multi_class='multinomial')
reg3.fit(X, y)

print(reg3.score(X, y))
plot_decision_regions(X, y, reg3)
plt.show()

