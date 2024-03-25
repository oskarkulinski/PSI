import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf

# Zadanie 0 - bez numeru

df_adv = pd.read_csv('https://raw.githubusercontent.com/przem85/bootcamp/master/statistics/Advertising.csv', index_col=0)
df_adv.head()

sns.pairplot(df_adv)
plt.show()

sns.heatmap(df_adv.corr(), annot=True)
plt.show()


