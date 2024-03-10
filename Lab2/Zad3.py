import numpy as np

x = np.random.randint(5, 15, size=100)
print(x)

counted = np.bincount(x)

print(counted)

print(counted.argmax())