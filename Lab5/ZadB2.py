import statsmodels.formula.api as smf
import pandas as pd


df_adv = pd.read_csv('https://raw.githubusercontent.com/przem85/bootcamp/master/statistics/Advertising.csv', index_col=0)
df_adv.head()

# est = smf.ols(formula='sales ~ I(newspaper)*I(TV)*I(radio)', data=df_adv).fit()

est = smf.ols(formula='sales ~ I(newspaper)*I(TV)*I(radio)', data=df_adv).fit()
print((est.summary2()))

est = smf.ols(formula='sales ~ I(TV) + I(radio) + I(TV**2) +  I(TV):I(radio)  + I(TV**3) + I(TV**4) + I(TV**5) + I(TV**6) ', data=df_adv).fit()
print((est.summary2()))

# formula: response ~ predictor + predictor
est = smf.ols(formula='sales ~ I(newspaper)+I(TV):I(radio)+np.log(radio+1)', data=df_adv).fit()
print((est.summary2()))

my_est = smf.ols(formula='', data=df_adv).fit()
print((my_est.summary2()))
