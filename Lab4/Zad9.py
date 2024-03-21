import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy import optimize
import math


def split_gaussian(x, m, sigma, tau):
    c = ((2 / np.pi) ** 0.5) / (sigma * (1 + tau))
    if x <= m:
        result = c * math.exp(-(x - m) ** 2 / (2 * sigma ** 2))
    else:
        result = c * math.exp(-(x - m) ** 2 / (2 * sigma ** 2 * tau ** 2))
    return result


def gauss_vec(x, m, sigma, tau):
    return [split_gaussian(a, m, sigma, tau) for a in x]


mu1 = 0
sigma1 = 1
tau1 = 1
mu2 = 0
sigma2 = 1
tau2 = 0.5
mu3 = 1
sigma3 = 0.5
tau3 = 1

x = np.linspace(-5, 5, 1000)

plt.plot(x, gauss_vec(x, mu1, sigma1, tau1), 'k-', lw=2, label='Rozkład normalny 1')
plt.show()
plt.plot(x, gauss_vec(x, mu2, sigma2, tau2), 'k-', lw=2, label='Rozkład normalny 2')
plt.show()
plt.plot(x, gauss_vec(x, mu3, sigma3, tau3), 'k-', lw=2, label='Rozkład normalny 3')
plt.show()

