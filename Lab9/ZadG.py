import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=1, learning_rate=0.5,
    algorithm="SAMME.R", random_state=42)
ada_clf.fit(X_train, y_train)
print(ada_clf.score(X_test, y_test))

plot_decision_regions(X, y, ada_clf)
plt.show()

ada_clf2 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=2, learning_rate=0.5,
    algorithm="SAMME.R", random_state=42)
ada_clf2.fit(X_train, y_train)
print(ada_clf2.score(X_test, y_test))
plot_decision_regions(X, y, ada_clf2)
plt.show()


ada_clf3 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=2, learning_rate=1,
    algorithm="SAMME.R", random_state=42)
ada_clf3.fit(X_train, y_train)
print(ada_clf3.score(X_test, y_test))
plot_decision_regions(X, y, ada_clf3)
plt.show()


ada_clf4 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=10, learning_rate=0.5,
    algorithm="SAMME.R", random_state=42)
ada_clf4.fit(X_train, y_train)
print(ada_clf4.score(X_test, y_test))
plot_decision_regions(X, y, ada_clf4)
plt.show()


ada_clf5 = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1),
    n_estimators=10, learning_rate=1,
    algorithm="SAMME.R", random_state=42)
ada_clf5.fit(X_train, y_train)
print(ada_clf5.score(X_test, y_test))
plot_decision_regions(X, y, ada_clf5)
plt.show()
