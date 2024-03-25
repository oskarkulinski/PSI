import statsmodels.api as sm
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn.linear_model as lm
import matplotlib.pyplot as plt

f = lambda x: ((x*2-1)*(x**2-2)*(x-2)+3)

x = stats.uniform(0,3).rvs(100)
y = f(x) + stats.norm(0,0.2).rvs(len(x))

M1 = np.vstack( (np.ones_like(x), x) ).T
p1 = np.linalg.lstsq(M1, y, rcond=None)

M2 = np.vstack( (np.ones_like(x), x, x**2) ).T
p2 = np.linalg.lstsq(M2, y, rcond=None)

Res1 = sm.OLS(y, M1).fit()
Res2 = sm.OLS(y, M2).fit()

print(Res1.summary2())
print(".....................................")
print(".....................................")
print(".....................................")
print(Res2.summary2())

import statsmodels.formula.api as smf
# Turn the data into a pandas DataFrame, so that we
# can address them in the formulas with their name
df = pd.DataFrame({'x':x, 'y':y})

# Fit the models, and show the results
Res1F = smf.ols('y~x', df).fit()
Res2F = smf.ols('y ~ x+I(x**2)', df).fit()

print(Res1F.summary())
print(".....................................")
print(".....................................")
print(".....................................")
print(Res2F.summary())