from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score

from sklearn import datasets
cancer = datasets.load_breast_cancer()

X = cancer.data
y = cancer.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

param_grid = {
    'max_depth': [3, 5, 8, 10],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 150, 200, 400],
    'gamma': [0, 0.5, 1, 2],
    'colsample_bytree': [1, 0.8, 0.5],
    'subsample': [1, 0.8, 0.5],
    'min_child_weight': [1, 5, 10]
}

from scipy.stats.distributions import uniform, randint

uniform_23 = uniform(2, 3)

# randint(low, high) losuje od low do high-1 włącznie!
randint_25 = randint(2, 6)

param_grid = {
    'max_depth': [3, 5, 8, 10],
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 150, 200, 400],
    'gamma': [0, 0.5, 1, 2],
    'colsample_bytree': [1, 0.8, 0.5],
    'subsample': [1, 0.8, 0.5],
    'min_child_weight': [1, 5, 10]
}

param_distribution = {
    'max_depth': randint(3, 11),
    'learning_rate': uniform(0.001, 0.1-0.001),
    'n_estimators': randint(50, 400),
    'gamma': uniform(0,2),
    'colsample_bytree': uniform(0.5, 0.5),
    'subsample': uniform(0.5, 0.5),
    'min_child_weight': randint(1, 11)
}

rnd = RandomizedSearchCV(XGBClassifier(), param_distribution, n_iter=100, cv=5, random_state=42, n_jobs=-1)
grd = GridSearchCV(XGBClassifier(), param_grid, cv=5, n_jobs=-1)

grd.fit(X_train, y_train)
rnd.fit(X_train, y_train)

print("Grid")
print(grd.score(X_test, y_test))
print("Random")
print(rnd.score(X_test, y_test))
