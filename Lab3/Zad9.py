import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.mlab as mlab
from matplotlib import colormaps
from scipy.stats import multivariate_normal

mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]


normal = multivariate_normal(mean, cov)

x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))

density = normal.pdf(pos)


fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(x, y, density, cmap='viridis', edgecolor='green')
plt.title("Density")

plt.show()

ax = plt.axes(projection='3d')
ax.contourf(x, y, density, cmap=colormaps['viridis'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Density Contours of 2D Normal Distribution')
plt.grid(True)
plt.show()
# Generate random samples from the normal distribution
random_samples = normal.rvs(size=1000)
# Plot the random samples
plt.scatter(random_samples[:, 0], random_samples[:, 1], s=10, color='red', alpha=0.3, label='Random Samples')

plt.legend()
plt.show()
