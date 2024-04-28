import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from mlxtend.plotting import plot_decision_regions
from sklearn.svm import SVC

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=.1, random_state=42)


plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


# coef0 : float, optional (default=0.0)
# Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

poly_kernel_svm_clf = SVC(kernel="poly", degree=1, coef0=0, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()


# coef0 : float, optional (default=0.0)
# Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

poly_kernel_svm_clf = SVC(kernel="poly", degree=2, coef0=0, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()


# coef0 : float, optional (default=0.0)
# Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

poly_kernel_svm_clf = SVC(kernel="poly", degree=3, coef0=0, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()


# coef0 : float, optional (default=0.0)
# Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.

poly_kernel_svm_clf = SVC(kernel="poly", degree=3, coef0=1, C=1)

poly_kernel_svm_clf.fit(X, y)
plot_decision_regions(X, y, poly_kernel_svm_clf)
plt.show()