import random

random.seed()

in_circle = 0
attempts = 10000000

for _ in range(attempts):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    if x**2 + y**2 <= 1:
        in_circle += 1


approximated_pi = 4 * in_circle/attempts

print(approximated_pi)