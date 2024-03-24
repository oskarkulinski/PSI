from sklearn import datasets
import numpy as np
from scipy import optimize
from Zad10 import pdf, log_likelihood
from Zad4 import mle

california_housing = datasets.fetch_california_housing()
data = california_housing.data[:, 7]

mu = np.mean(data)
sigma = np.std(data)
tau = np.std(data)


def normal(params, x):
    muP, sigmaP = params
    return mle(x, muP, sigmaP)


def split(params, x):
    muP, sigmaP, tauP = params
    return log_likelihood(x, muP, sigmaP, tauP)


res1 = optimize.fmin_cg(normal, np.array([mu, sigma]), args=[data])
print(res1)
print()

res2 = optimize.fmin_cg(split, np.array([mu, sigma, tau]), args=[data])
print(res2)
print()

