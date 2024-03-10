import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

inFile = 'http://ww2.amstat.org/publications/jse/datasets/babyboom.dat.txt'
data = pd.read_csv(inFile, sep='[ ]+', header=None, engine='python',names= ['sex', 'Weight', 'Minutes'])
df = data[['Minutes', 'sex', 'Weight']]

m = data[data['sex'] == 1]
s = data[data['sex'] == 2]

sw = s.Weight.values
mw = m.Weight.values
plt.scatter(np.arange(len(sw)), sw, label='female', c='pink')
plt.scatter(np.arange(len(mw)), mw, label='male', c='blue')
plt.legend()
plt.show()

plt.hist(sw, bins=25, label='female', color='pink')
plt.hist(mw, bins=25, label='male', color='blue')
plt.legend()
plt.show()


sns.kdeplot(sw, label='female', color='pink')
sns.kdeplot(mw, label='male', color='blue')
plt.legend()
plt.show()


plt.plot(stats.cumfreq(sw,numbins=25)[0], label='female', c='pink')
plt.plot(stats.cumfreq(mw,numbins=25)[0], label='male', c='blue')
plt.legend()
plt.show()

plt.boxplot([sw, mw], sym='*', labels=['female', 'male'])
plt.legend()
plt.show()

sns.violinplot([sw, mw])
#sns.violinplot(mw, label='male', color='blue')
plt.show()
