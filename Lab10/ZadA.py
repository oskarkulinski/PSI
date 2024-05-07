import numpy as np
import sklearn.datasets
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

#low noise, plenty of samples, should be easy
X, y = sklearn.datasets.make_moons(n_samples=1000, noise=.05)

import matplotlib.pyplot as plt

plt.scatter(X[:,0],X[:,1],c=y)
plt.axis('equal')
plt.show()

iris = load_iris()
X = iris.data[:, (2, 3)]  # petal length, petal width
y = (iris.target == 0).astype(np.int32)

per_clf = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])


from mlxtend.plotting import plot_decision_regions

from sklearn.linear_model import Perceptron
per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)


plot_decision_regions(X, y, clf=per_clf, zoom_factor=0.2)

#  IRIS

iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

per_clf = Perceptron(random_state=42)
per_clf.fit(X_train, y_train)


plot_decision_regions(X, y, clf=per_clf, zoom_factor=0.2)
from sklearn.metrics import f1_score

pred = per_clf.predict(X_test)
print(f1_score(y_test, y_pred))