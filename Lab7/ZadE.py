import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
list(data.keys())

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'], random_state=0)

logreg = LogisticRegression(max_iter=10000)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_train)
print('On train')
print(metrics.precision_score(y_train, y_pred))
print(metrics.recall_score(y_train, y_pred))
print(metrics.f1_score(y_train, y_pred))

# na test
y_pred = logreg.predict(X_test)
print("On test")
print(metrics.precision_score(y_test, y_pred))
print(metrics.recall_score(y_test, y_pred))
print(metrics.f1_score(y_test, y_pred))