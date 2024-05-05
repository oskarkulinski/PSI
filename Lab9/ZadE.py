import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())

X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"]).astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.show()

plt.hist(y_train)
plt.hist(y_test)
plt.show()

from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_decision_regions

clf1 = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
clf2 = RandomForestClassifier(n_estimators=50, max_leaf_nodes=2, n_jobs=-1, random_state=42)
clf3 = RandomForestClassifier(n_estimators=5, max_leaf_nodes=2, n_jobs=-1, random_state=42)


clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)

print(clf1.score(X_test, y_test))
print(clf2.score(X_test, y_test))
print(clf3.score(X_test, y_test))

plot_decision_regions(X_test, y_test, clf1)
plt.show()

plot_decision_regions(X_test, y_test, clf2)
plt.show()

plot_decision_regions(X_test, y_test, clf3)
plt.show()


