import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
data= iris.data

df = pd.DataFrame(data)
corr = df.corr()


sns.pairplot(df)
plt.show()


mask = np.zeros_like(corr, dtype=np.bool_)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")

sns.heatmap(df)
plt.show()