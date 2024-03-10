import numpy as np
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)
print(X.shape)
print(y.shape)

y = np.where(y == 0, -1, y)

X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))

print(X)
print()
print(y)