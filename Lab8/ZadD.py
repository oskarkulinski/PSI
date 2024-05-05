import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

np.random.seed(1)
cancer = datasets.load_breast_cancer()
# print description
print(cancer.DESCR)

# get the data
X = cancer.data
y = cancer.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

plt.hist(y_train, alpha=0.5)
plt.hist(y_test, alpha=0.5)
plt.show()

from sklearn.model_selection import StratifiedKFold

seed = 123
kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

pipe1 = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC(kernel='rbf'))])
pipe2 = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC(kernel='poly'))])
pipe3 = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC(kernel='linear'))])
pipe4 = Pipeline([('preprocessing', StandardScaler()), ('classifier', LogisticRegression())])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_1 = GridSearchCV(pipe1, param_grid, cv=kfold, return_train_score=True)

grid_1.fit(X_train, y_train)

grid_2 = GridSearchCV(pipe2, param_grid, cv=kfold, return_train_score=True)

grid_2.fit(X_train, y_train)

grid_3 = GridSearchCV(pipe3, param_grid, cv=kfold, return_train_score=True)

grid_3.fit(X_train, y_train)

grid_4 = GridSearchCV(pipe4, param_grid, cv=kfold, return_train_score=True)

grid_4.fit(X_train, y_train)


def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f"):
    if ax is None:
        ax = plt.gca()
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(), img.get_facecolors(),
                               img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center")
    return img


import pandas as pd

# convert to DataFrame
results = pd.DataFrame(grid_1.cv_results_)
# show the first 5 rows
# display(results.head())

len(results.mean_test_score)

scores = np.array(results.mean_test_score).reshape(6, 6, 2)
scores = scores[:, :, 0]
# plot the mean cross-validation scores
heatmap(scores, xlabel='classifier__gamma', xticklabels=param_grid['classifier__gamma'], ylabel='classifier__C',
        yticklabels=param_grid['classifier__C'], cmap="viridis")
plt.show()

scores = np.array(results.mean_test_score).reshape(6, 6, 2)
scores = scores[:, :, 1]
# plot the mean cross-validation scores
heatmap(scores, xlabel='classifier__gamma', xticklabels=param_grid['classifier__gamma'], ylabel='classifier__C',
        yticklabels=param_grid['classifier__C'], cmap="viridis")
plt.show()

from sklearn import metrics

models = []
models.append(('SVM rbf', grid_1.best_estimator_))
models.append(('SVM poly', grid_2.best_estimator_))
models.append(('SVM linear', grid_3.best_estimator_))
models.append(('SVM logistic', grid_4.best_estimator_))

precision_score = []
recall_score = []
f1_score = []
accuracy_score = []
for name, model in models:
    print(name)
    print("R^2: {}".format(metrics.precision_score(y_test, model.predict(X_test))))
    print("recall_score: {}".format(metrics.recall_score(y_test, model.predict(X_test))))
    print("f1_score: {}".format(metrics.f1_score(y_test, model.predict(X_test))))
    print("accuracy_score: {}".format(metrics.accuracy_score(y_test, model.predict(X_test))))
    precision_score.append(metrics.precision_score(y_test, model.predict(X_test)))
    recall_score.append(metrics.recall_score(y_test, model.predict(X_test)))
    f1_score.append(metrics.f1_score(y_test, model.predict(X_test)))
    accuracy_score.append(metrics.accuracy_score(y_test, model.predict(X_test)))

import pandas as pd

d = {'precision_score': precision_score,
     'recall_score': recall_score,
     'f1_score': f1_score,
     'accuracy_score': accuracy_score
     }
df = pd.DataFrame(data=d)
df.insert(loc=0, column='Method', value=np.array(['SVM rbf', 'SVM poly', 'SVM linear', 'Logistic']))
print(df)
