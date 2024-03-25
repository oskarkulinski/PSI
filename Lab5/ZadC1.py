import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import model_selection
from sklearn import  metrics
from statsmodels.formula.api import ols

data_url = "http://lib.stat.cmu.edu/datasets/boston"
boston = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([boston.values[::2, :], boston.values[1::2, :2]])
target = boston.values[1::2, 2]

print(boston.keys())
print(boston.shape)

bos=pd.DataFrame(boston)
bos.head()

bos=pd.DataFrame(data)
feature_name = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO','B', 'LSTAT']
bos.columns = feature_name

print(target[0:5])
bos['PRICE']=target
bos.head()

model = ols("PRICE ~ CRIM + ZN + I(2**INDUS) + ZN:INDUS+ I(INDUS ** 2.0)", bos).fit()
# Print the summary
print((model.summary()))

metrics.r2_score(bos.PRICE, model.predict(bos))

my_est = ols(formula="PRICE~ I(CRIM) + I(CRIM**2) + I(ZN) + I(ZN**2)"
                     " + I(INDUS) + I(CHAS) + " 
                     "I(NOX) + I(NOX**2) + I(RM) + I(RM**2) + I(AGE)  + "
                     "I(AGE**2) + I(DIS) + I(DIS**2) + " 
                     "I(RAD) + I(RAD**2) + I(TAX) + I(TAX**2) + I(PTRATIO) + I(PTRATIO**2) + I(B) + "
                     "I(B**2) + I(LSTAT) + I(LSTAT**2) + I(LSTAT**3)", data=bos).fit()
print(my_est.summary())
print(metrics.r2_score(bos.PRICE, my_est.predict(bos)))
# Formula          R-squared
# Sum all with Rm^2 0.803
# + Rm^2, Dist^2    0.809
# + Age^2           0.810
# - CHAS            0.806
# + Rad^2 + CHAS    0.810
# + Nox^2           0.810
# + LSTAT^2         0.823
# + B^2             0.824
# + Lstat^3         0.826
# + Ptratio^2       0.828
# + Crim^2          0.831
# + ZN^2            0.832
# - NOX - NOX^2     0.815
# + Tax^2 + NOXes   0.833