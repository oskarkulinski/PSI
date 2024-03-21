import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy import optimize
import math


# split normal distribution pdf
def Gpdf(x, mu, sigma):
    return 1 / (sigma * (2 * np.pi) ** .5) * np.e ** (-(x - mu) ** 2 / (2 * sigma ** 2))


def lpdf(x, mu, sigma):
    return (-(x - mu) ** 2 / (2 * sigma ** 2)) * (math.log(np.e, 1 / (sigma * (2 * np.pi) ** .5)) + 1)


def mle(x, mu, sigma):
    mle = 0
    for i in range(len(x)):
        mle += lpdf(x[i], mu, sigma)
    return mle


mu1 = 0
sigma1 = 1
mu2 = 0
sigma2 = 2
mu3 = 1
sigma3 = 1
mu4 = 0.5
sigma4 = 0.2

x = np.linspace(1, 200)

sample = stats.norm.rvs(x, 1000)

print(mle(sample, mu1, sigma1))
print(mle(sample, mu2, sigma2))
print(mle(sample, mu3, sigma3))
print(mle(sample, mu4, sigma4))
