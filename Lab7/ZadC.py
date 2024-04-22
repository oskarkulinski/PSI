import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
list(data.keys())


from sklearn.linear_model import LogisticRegression

X, y = data.data, data.target
print(X.shape)
print(y.shape)

plt.figure(figsize=(7, 7))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X, y)

print(log_reg.intercept_)
print(log_reg.coef_)
print()

(log_reg.predict(X))

print(log_reg.predict_proba(X))

print(metrics.accuracy_score(log_reg.predict(X),y))


from sklearn.model_selection import train_test_split
X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.20)


log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

print(log_reg.intercept_)
print(log_reg.coef_)
print()

print((log_reg.predict(X_train)))

print(log_reg.predict_proba(X_train))

print(metrics.accuracy_score(log_reg.predict(X_test),y_test))

