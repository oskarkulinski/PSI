import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
wine = load_wine()
print(list(wine.target_names))
print(wine.data)
wine.target[ wine.target ==0 ] = 1 # we use only two classes
print(wine.target)


X_train, X_test, y_train, y_test = train_test_split( wine.data, wine.target, stratify = wine.target, random_state=42)
print(X_train.shape)

# C = 1

log_reg_1 = LogisticRegression(C=1, random_state=42)
log_reg_1.fit(X_train, y_train)

y_proba_1 = log_reg_1.predict_proba(X_test)

# C = 100

log_reg_2 = LogisticRegression(C=100, random_state=42)
log_reg_2.fit(X_train, y_train)

y_proba_2 = log_reg_2.predict_proba(X_train)


# C = 100

log_reg_3 = LogisticRegression(C=0.01, random_state=42)
log_reg_3.fit(X_train, y_train)

y_proba_3 = log_reg_3.predict_proba(X_test)


print(log_reg_1.coef_)
print(log_reg_2.coef_)
print(log_reg_3.coef_)

plt.figure(figsize=(10,6))
plt.scatter(np.arange(1, len(log_reg_1.coef_[0]) + 1), log_reg_1.coef_[0], label='1')
plt.scatter(np.arange(1, len(log_reg_2.coef_[0]) + 1), log_reg_2.coef_[0], label='2')
plt.scatter(np.arange(1, len(log_reg_3.coef_[0]) + 1), log_reg_3.coef_[0], label='3')
plt.title("Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()