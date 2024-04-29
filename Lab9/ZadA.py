import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

plt.figure(figsize=(10, 4))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42, criterion='entropy')
tree_clf.fit(X, y)

from sklearn.tree import export_graphviz

export_graphviz(
    tree_clf,
    out_file="./iris_tree1.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)


from matplotlib.colors import ListedColormap

def plot_decision_boundary(clf, X, y, axes=[0, 7.5, 0, 3], iris=True, legend=False, plot_training=True):
    x1s = np.linspace(axes[0], axes[1], 100)
    x2s = np.linspace(axes[2], axes[3], 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    if not iris:
        custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
        plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=0.8)
    if plot_training:
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
        plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
        plt.axis(axes)
    if iris:
        plt.xlabel("Petal length", fontsize=14)
        plt.ylabel("Petal width", fontsize=14)
    else:
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
    if legend:
        plt.legend(loc="lower right", fontsize=14)

plt.figure(figsize=(8, 4))
plot_decision_boundary(tree_clf, X, y)
plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
plt.plot([4.95, 4.95], [0, 1.75], "k:", linewidth=2)
plt.plot([4.85, 4.85], [1.75, 3], "k:", linewidth=2)
# plt.text(1.40, 1.0, "Depth=0", fontsize=15)
# plt.text(3.2, 1.80, "Depth=1", fontsize=13)
# plt.text(4.05, 0.5, "(Depth=2)", fontsize=11)

plt.show()

tree_clf.predict_proba([[5, 1.5]])


tree_clf.predict([[5, 1.5]])

from mlxtend.plotting import plot_decision_regions

plt.figure(figsize=(10, 5))
plot_decision_regions(X, y, tree_clf)
plt.show()

# Tree 1
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=.1, random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

tree_clf = DecisionTreeClassifier(max_depth=10, random_state=42)

tree_clf.fit(X, y)

y_pred1 = tree_clf.predict(X)

plt.figure(figsize=(10, 6))
plot_tree(tree_clf, filled=True, rounded=True, class_names=['0', '1'], feature_names=['Feature 1', 'Feature 2'])
plt.show()

plt.figure(figsize=(8, 4))
plot_decision_regions(X, y, tree_clf)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Tree 2
deep_tree_clf2 = DecisionTreeClassifier(min_samples_leaf=4, random_state=42)

deep_tree_clf2.fit(X, y)

y_pred2 = deep_tree_clf2.predict(X)

plt.figure(figsize=(10, 6))
plot_tree(deep_tree_clf2, filled=True, rounded=True, class_names=['0', '1'], feature_names=['Feature 1', 'Feature 2'])
plt.show()

plt.figure(figsize=(8, 4))
plot_decision_regions(X, y, deep_tree_clf2)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

from sklearn.metrics import precision_score, recall_score,f1_score, accuracy_score,roc_auc_score, roc_curve
# Porownanie

precision1 = precision_score(y, y_pred1)
recall1 = recall_score(y, y_pred1)
f11 = f1_score(y, y_pred1)
accuracy1 = accuracy_score(y, y_pred1)
roc_auc1 = roc_auc_score(y, y_pred1)

print("Precision treee 1:", precision1)
print("Recall tree 1:", recall1)
print("F1-score tree 1:", f11)
print("Accuracy tree 1:", accuracy1)
print("ROC AUC score tree 1:", roc_auc1)
print()

precision2 = precision_score(y, y_pred2)
recall2 = recall_score(y, y_pred2)
f12 = f1_score(y, y_pred2)
accuracy2 = accuracy_score(y, y_pred2)
roc_auc2 = roc_auc_score(y, y_pred2)

print("Precision treee 2:", precision2)
print("Recall tree 2:", recall2)
print("F1-score tree 2:", f12)
print("Accuracy tree 2:", accuracy2)
print("ROC AUC score tree 2:", roc_auc2)

# Narysowanie krzywej ROC
fpr1, tpr1, thresholds1 = roc_curve(y, y_pred1)
fpr2, tpr2, thresholds2 = roc_curve(y, y_pred2)

plt.figure(figsize=(8, 6))
plt.plot(fpr1, tpr1, label='ROC curve 1(area = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, label='ROC curve 2(area = %0.2f)' % roc_auc2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
