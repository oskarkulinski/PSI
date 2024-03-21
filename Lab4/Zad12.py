from sklearn import datasets
import numpy as np
from Zad10 import pdf, log_likelihood
from Zad4 import mle

california_housing = datasets.fetch_california_housing()
data=california_housing.data[:, 7]


mu = np.mean(data)
sigma = np.std(data)
tau = np.std(data)
