from scipy import optimize
import numpy as np

f = lambda x, y: (x + 1) ** 2 + y ** 2

res = optimize.fmin_cg(f, np.array((0, 0)))

print(res)

