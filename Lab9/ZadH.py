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


from sklearn.ensemble import GradientBoostingClassifier

grd_clf1 = GradientBoostingClassifier(n_estimators=1, learning_rate=0.5, random_state=42)
grd_clf1.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, grd_clf1)
plt.show()

grd_clf2 = GradientBoostingClassifier(n_estimators=2, learning_rate=0.5, random_state=42)
grd_clf2.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, grd_clf2)
plt.show()

grd_clf3 = GradientBoostingClassifier(n_estimators=2, learning_rate=1, random_state=42)
grd_clf3.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, grd_clf3)
plt.show()

grd_clf4 = GradientBoostingClassifier(n_estimators=10, learning_rate=0.5, random_state=42)
grd_clf4.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, grd_clf4)
plt.show()

grd_clf5 = GradientBoostingClassifier(n_estimators=10, learning_rate=1, random_state=42)
grd_clf5.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, grd_clf5)
plt.show()

try:
    import xgboost
except ImportError as ex:
    print("Error: the xgboost library is not installed.")
    xgboost = None

xgb_clf2 = xgboost.XGBClassifier(n_estimators=2, learning_rate=0.5, random_state=42)
xgb_clf2.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, xgb_clf2, colors='green,red')
plt.show()

xgb_clf3 = xgboost.XGBClassifier(n_estimators=2, learning_rate=1, random_state=42)
xgb_clf3.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, xgb_clf3, colors='green,red')
plt.show()

xgb_clf4 = xgboost.XGBClassifier(n_estimators=10, learning_rate=0.5, random_state=42)
xgb_clf4.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, xgb_clf4, colors='green,red')
plt.show()

xgb_clf5 = xgboost.XGBClassifier(n_estimators=10, learning_rate=1, random_state=42)
xgb_clf5.fit(X_train, y_train)

plot_decision_regions(X_train, y_train, xgb_clf5, colors='green,red')
plt.show()