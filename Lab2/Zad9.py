import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mean = 0
sigma = 1

x = np.linspace(mean - 4 * sigma, mean + 4 * sigma, 1000)
y = norm.pdf(x, mean, sigma)

plt.plot(x, y, label=f'Normal Distribution (mean={mean}, sigma={sigma})')

area = np.trapz(y, x)
print("Area under the curve:", area)


plt.fill_between(x, y, where=((x >= mean - 3 * sigma) & (x <= mean + 3 * sigma)), alpha=0.3, label='99.7% of data')


plt.legend()
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.title('Normal Distribution')


plt.show()

p_within_sigma = norm.cdf(mean + sigma, mean, sigma) - norm.cdf(mean - sigma, mean, sigma)
p_within_2sigma = norm.cdf(mean + 2 * sigma, mean, sigma) - norm.cdf(mean - 2 * sigma, mean, sigma)
p_within_3sigma = norm.cdf(mean + 3 * sigma, mean, sigma) - norm.cdf(mean - 3 * sigma, mean, sigma)

print("P(X in [mean - sigma, mean + sigma]):", p_within_sigma)
print("P(X in [mean - 2sigma, mean + 2sigma]):", p_within_2sigma)
print("P(X in [mean - 3sigma, mean + 3sigma]):", p_within_3sigma)
