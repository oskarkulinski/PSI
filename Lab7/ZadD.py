import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

np.random.seed(42)

from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='liac-arff')

# X[1]
X = X / 255.

print(y)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
for i, ax in enumerate(axes.ravel()):
    ax.imshow(X[i].reshape(28, 28), cmap=plt.cm.gist_gray)
    ax.set_title(y[i])  # Assuming y contains labels for each image

plt.show()
plt.show()

import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    import matplotlib
    plt.imshow(image, cmap=matplotlib.cm.binary, **options)
    plt.axis("off")


plt.figure(figsize=(9, 9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
plt.show()

# X_train = mnist.train.images
# y_train = mnist.train.labels
# X_test = mnist.test.images
# y_test = mnist.test.labels

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(y_train.shape)

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

print(X_train.shape)
print(y_train_5.shape)
print(np.unique(y_train_5))

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train_5)

clf.predict([some_digit])

from sklearn.model_selection import cross_val_score

cross_val_score(clf, X_train, y_train_5, cv=3, scoring="accuracy")

from sklearn.base import BaseEstimator


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return -5


never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(clf, X_train, y_train_5, cv=3)
print(y_train_pred)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)

y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)

from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))

from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

from sklearn.metrics import classification_report
print(classification_report(y_train_5, y_train_pred))#, target_names=["not 5", "5"]))



never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(clf, X_test, y_test, cv=3)
print(y_train_pred)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)

y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)

from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))

from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)

from sklearn.metrics import classification_report
print(classification_report(y_train_5, y_train_pred))#, target_names=["not 5", "5"]))