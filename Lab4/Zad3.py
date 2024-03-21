import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy import optimize

# Parametry rozkładu jednostajnego
a, b = -2, 4  # zakładamy, że chcemy losować z przedziału [-2, 4]

# Parametry rozkładu normalnego
mu, sigma = 3, 2

# Liczba próbek
N = 10000

# Losowanie próbki z rozkładu jednostajnego
uniform_data = np.random.uniform(a, b, N)

# Punkty, dla których będziemy rysować funkcję gęstości rozkładu normalnego
t = np.linspace(-3, 5, 1000)

mu2, sigma2 = stats.norm.fit(uniform_data)


plt.plot(t, stats.norm.pdf(t, mu2, sigma2), 'k-', lw=2, label='Rozkład normalny ($\mu=3$, $\sigma=2$)')


# Dodanie legendy i tytułu
plt.legend()
plt.show()

