import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy import optimize
import math

# Parametry rozkładu jednostajnego
a, b = -1.5, 1.5  # zakładamy, że chcemy losować z przedziału [-2, 4]

uniform_data = np.random.uniform(a, b, 100)

# Parametry rozkładu normalnego
mu = sum(uniform_data) / len(uniform_data)

sigma = sum((d - mu)**2 for d in uniform_data)/len(uniform_data)

# Liczba próbek
N = 100


# Punkty, dla których będziemy rysować funkcję gęstości rozkładu normalnego
t = np.linspace(-5, 5, 100)


# Rysowanie funkcji gęstości rozkładu normalnego
plt.plot(t, stats.norm.pdf(t, mu, sigma), 'k-', lw=2, label='Rozkład normalny ($\mu=1$, $\sigma=1$)')

# Dodanie legendy i tytułu
plt.legend()
plt.title('Porównanie próbki jednostajnej i rozkładu normalnego')
plt.show()

