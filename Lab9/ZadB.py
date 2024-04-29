import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

np.random.seed(1)
cancer = datasets.load_breast_cancer()
# print description
# print(cancer.DESCR)

# get the data
X = cancer.data
y = cancer.target


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

plt.hist(y_train, alpha=0.5)
plt.hist(y_test, alpha=0.5)
plt.show()

tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

tree = DecisionTreeClassifier(max_depth=4, random_state=0)

tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

plot_tree(tree)
plt.show()

plt.figure(figsize=(8,6))
plt.xlim([0, 1])
plt.bar( tree.feature_importances_, cancer.feature_names)
plt.show()