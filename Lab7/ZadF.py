import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())

X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int32)  # 1 if Iris-Virginica, else 0

plt.hist(X[y==0, :])
plt.hist(X[y==1, :])
plt.xlim(0,3)
plt.show()

#C = 1
from sklearn.linear_model import LogisticRegression
log_reg_1 = LogisticRegression(C = 1, random_state=42)
log_reg_1.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba_1 = log_reg_1.predict_proba(X_new)


plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot(X_new, y_proba_1[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba_1[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.show()


#C = 10

log_reg_2 = LogisticRegression(C = 10, random_state=42)
log_reg_2.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba_2 = log_reg_2.predict_proba(X_new)


plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot(X_new, y_proba_2[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba_2[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.show()

# C = 100

log_reg_3 = LogisticRegression(C = 100, random_state=42)
log_reg_3.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba_3 = log_reg_3.predict_proba(X_new)


plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot(X_new, y_proba_3[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba_3[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.show()


plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot(X_new, y_proba_1[:, 1], "g-", linewidth=2, label="Iris-Virginica 1")
plt.plot(X_new, y_proba_1[:, 0], "b--", linewidth=2, label="Not Iris-Virginica 1")
plt.plot(X_new, y_proba_2[:, 1], "g-", linewidth=2, label="Iris-Virginica 2")
plt.plot(X_new, y_proba_2[:, 0], "b--", linewidth=2, label="Not Iris-Virginica 2")
plt.plot(X_new, y_proba_3[:, 1], "g-", linewidth=2, label="Iris-Virginica 3")
plt.plot(X_new, y_proba_3[:, 0], "b--", linewidth=2, label="Not Iris-Virginica 3")
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
plt.show()

# 2D example ==========================================

X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.int32)

# C = 1

log_reg_1 = LogisticRegression(C=1, random_state=42)
log_reg_1.fit(X, y)

x0, x1 = np.meshgrid(
    np.linspace(2.9, 7, 500).reshape(-1, 1),
    np.linspace(0.8, 2.7, 200).reshape(-1, 1),
)
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba_1 = log_reg_1.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba_1[:, 1].reshape(x0.shape)
contour1 = plt.contour(x0, x1, zz, cmap=plt.cm.brg)


left_right = np.array([2.9, 7])
boundary1 = -(log_reg_1.coef_[0][0] * left_right + log_reg_1.intercept_[0]) / log_reg_1.coef_[0][1]

plt.clabel(contour1, inline=1, fontsize=12)
plt.plot(left_right, boundary1, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])

plt.show()

# C = 10

log_reg_2 = LogisticRegression(C=10, random_state=42)
log_reg_2.fit(X, y)

x0, x1 = np.meshgrid(
    np.linspace(2.9, 7, 500).reshape(-1, 1),
    np.linspace(0.8, 2.7, 200).reshape(-1, 1),
)
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba_2 = log_reg_2.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba_2[:, 1].reshape(x0.shape)
contour2 = plt.contour(x0, x1, zz, cmap=plt.cm.brg)


left_right = np.array([2.9, 7])
boundary2 = -(log_reg_2.coef_[0][0] * left_right + log_reg_2.intercept_[0]) / log_reg_2.coef_[0][1]

plt.clabel(contour2, inline=1, fontsize=12)
plt.plot(left_right, boundary2, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])

plt.show()

# C = 100

log_reg_3 = LogisticRegression(C=100, random_state=42)
log_reg_3.fit(X, y)

x0, x1 = np.meshgrid(
    np.linspace(2.9, 7, 500).reshape(-1, 1),
    np.linspace(0.8, 2.7, 200).reshape(-1, 1),
)
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba_3 = log_reg_3.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba_3[:, 1].reshape(x0.shape)
contour3 = plt.contour(x0, x1, zz, cmap=plt.cm.brg)


left_right = np.array([2.9, 7])
boundary3 = -(log_reg_3.coef_[0][0] * left_right + log_reg_3.intercept_[0]) / log_reg_3.coef_[0][1]

plt.clabel(contour3, inline=1, fontsize=12)
plt.plot(left_right, boundary3, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])

plt.show()


plt.clabel(contour1, inline=1, fontsize=12)
plt.clabel(contour2, inline=1, fontsize=12)
plt.clabel(contour3, inline=1, fontsize=12)
plt.plot(left_right, boundary1, "k--", linewidth=3)
plt.plot(left_right, boundary2, "k--", linewidth=3)
plt.plot(left_right, boundary3, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])

plt.show()
