import numpy as np
from scipy import optimize


def pdf(x, mu, sigma, tau):
    c = ((2 / np.pi) ** 0.5) / (sigma + tau)
    if x < mu:
        return c * np.exp((-(x - mu) ** 2) / (2 * sigma ** 2))
    else:
        return c * np.exp((-(x - mu) ** 2) / (2 * tau ** 2))


def log_likelihood(x, mu, sigma, tau):
    return sum(np.log(pdf(el, mu, sigma, tau)) for el in x)


mu1 = 0
sigma1 = 1
tau1 = 1

x0 = np.asarray((0, 0))  # Initial guess.
res1 = optimize.fmin_cg(log_likelihood, x0, args=(mu1, sigma1, tau1))
print(res1)
print()
