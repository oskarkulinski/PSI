import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

dataset = pd.read_csv('diabetes.csv')
X = dataset.drop(axis="columns", labels='Outcome')
Y = dataset['Outcome']

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

from sklearn.model_selection import StratifiedKFold

seed=123
kfold = StratifiedKFold(n_splits=5)

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
#### SVC
pipe1 = Pipeline([('preprocessing', StandardScaler()), ('classifier', LinearSVC(C=1))])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100]
}

grid_1 = GridSearchCV(pipe1, param_grid, cv=kfold, return_train_score=True)

grid_1.fit(X_train, y_train)
print("LinearSVC: ", accuracy_score(grid_1.predict(X_test), y_test))

#### LogisticRegression

pipe2 = Pipeline([('preprocessing', StandardScaler()), ('classifier', LogisticRegression(C=1))])

grid_2 = GridSearchCV(pipe2, param_grid, cv=kfold, return_train_score=True)

grid_2.fit(X_train, y_train)
print("LogisticRegression: ", accuracy_score(grid_2.predict(X_test), y_test))

#### SVC

pipe3 = Pipeline([('preprocessing', StandardScaler()), ('classifier', SVC(C=1))])

grid_3 = GridSearchCV(pipe3, param_grid, cv=kfold, return_train_score=True)

grid_3.fit(X_train, y_train)
print("SVC: ", accuracy_score(grid_3.predict(X_test), y_test))

#### KNeighbors

pipe4 = Pipeline([('preprocessing', StandardScaler()), ('classifier', KNeighborsClassifier(n_neighbors=1))])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_neighbors': [1, 2, 3, 4, 5, 10, 100]
}

grid_4 = GridSearchCV(pipe4, param_grid, cv=kfold, return_train_score=True)

grid_4.fit(X_train, y_train)
print("Kneigbors: ", accuracy_score(grid_4.predict(X_test), y_test))

#### DecisionTree

pipe5 = Pipeline([('preprocessing', StandardScaler()), ('classifier', DecisionTreeClassifier())])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__max_depth': [1, 2, 3, 4, 5, 10, 20]
}

grid_5 = GridSearchCV(pipe5, param_grid, cv=kfold, return_train_score=True)

grid_5.fit(X_train, y_train)
print("DecisionTree: ", accuracy_score(grid_5.predict(X_test), y_test))


#### RandomForest

pipe6 = Pipeline([('preprocessing', StandardScaler()), ('classifier', RandomForestClassifier())])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_estimators': [1, 2, 3, 5, 10, 15]
}

grid_6 = GridSearchCV(pipe6, param_grid, cv=kfold, return_train_score=True)

grid_6.fit(X_train, y_train)
print("RandomForest: ", accuracy_score(grid_6.predict(X_test), y_test))

#### Bagging

pipe7 = Pipeline([('preprocessing', StandardScaler()), ('classifier', BaggingClassifier())])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_estimators': [1, 2, 3, 4, 5, 10, 15]
}

grid_7 = GridSearchCV(pipe7, param_grid, cv=kfold, return_train_score=True)

grid_7.fit(X_train, y_train)
print("Bagging: ", accuracy_score(grid_7.predict(X_test), y_test))

#### Extra Trees

pipe8 = Pipeline([('preprocessing', StandardScaler()), ('classifier', ExtraTreesClassifier())])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_estimators': [1, 2, 3, 4, 5, 10, 15]
}

grid_8 = GridSearchCV(pipe8, param_grid, cv=kfold, return_train_score=True)

grid_8.fit(X_train, y_train)
print("ExtraTrees: ", accuracy_score(grid_8.predict(X_test), y_test))

#### AdaBoost

pipe9 = Pipeline([('preprocessing', StandardScaler()), ('classifier', AdaBoostClassifier())])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_estimators': [1, 2, 3, 4, 5, 10, 15]
}

grid_9 = GridSearchCV(pipe9, param_grid, cv=kfold, return_train_score=True)

grid_9.fit(X_train, y_train)
print("AdaBoost: ", accuracy_score(grid_9.predict(X_test), y_test))

#### GradientBoosting


pipe10 = Pipeline([('preprocessing', StandardScaler()), ('classifier', GradientBoostingClassifier())])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_estimators': [1, 2, 3, 4, 5, 10, 15]
}

grid_10 = GridSearchCV(pipe10, param_grid, cv=kfold, return_train_score=True)

grid_10.fit(X_train, y_train)
print("GradientBoosting: ", accuracy_score(grid_10.predict(X_test), y_test))

#### Voting

pipe11 = Pipeline([('preprocessing', StandardScaler()), ('classifier', VotingClassifier(
    estimators=[('Log', LogisticRegression()), ('LinSVC',LinearSVC()),
                ('Kneighbors',KNeighborsClassifier())]
))])

param_grid = {
    'preprocessing': [StandardScaler(), None],
}

grid_11 = GridSearchCV(pipe11, param_grid, cv=kfold, return_train_score=True)

grid_11.fit(X_train, y_train)
print("Voting: ", accuracy_score(grid_11.predict(X_test), y_test))

#### XGBClassifier


pipe12 = Pipeline([('preprocessing', StandardScaler()), ('classifier', XGBClassifier())])

param_grid = {
    'preprocessing': [StandardScaler(), None],
    'classifier__n_estimators': [1, 2, 3, 4, 5, 10, 15]
}

grid_12 = GridSearchCV(pipe12, param_grid, cv=kfold, return_train_score=True)

grid_12.fit(X_train, y_train)
print("LinearSVC: ", accuracy_score(grid_12.predict(X_test), y_test))
