import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

mean = [0.5, 0.5]  # Średnia próbki
cov = [[0.1, 0.05], [0.05, 0.2]]

normal = multivariate_normal(mean, cov)
sample = normal.rvs(1000)


sample_mean = np.mean(sample, axis=0)
sample_cov = np.cov(sample)

print(sample_mean)
print(sample_cov)


eigenvalue, eigenvectors = np.linalg.eig(sample_cov)

x, y = np.mgrid[0:1:0.01, 0:1:0.01]
pos = np.dstack((x, y))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
ax.scatter(sample[:, 0], sample[:, 1])
plt.contour(x, y, normal.pdf(pos), levels=10, cmap='viridis', alpha=0.5)
ax.scatter(eigenvectors[:, 0], eigenvectors[:, 1])
ax.set_xlim([0,1])
ax.set_ylim([0,1])
plt.show()
