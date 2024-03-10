import numpy as np

mean = [6, 1, 11, 2, 4]

cov = [[  6,  1,  -3,   0,   2],
       [  1,  10,   1,   0,  -1],
    [ -3,   1,   6 ,  1,   0],
[  0 ,  0,   1,   3 ,  1],
[  2,  -1 ,  0 ,  1 ,  4]]

points = np.random.multivariate_normal(mean=mean, cov=cov, size=100).transpose()

points = (points - np.mean(points, axis=1)) / np.std(points, axis=1)

print(points)
print()
print(np.mean(points))
print(np.cov(points))
