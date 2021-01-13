import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as scs
import statsmodels.api as sm

import matplotlib.pyplot as plt 
import seaborn as sns


# load data
data = pd.read_csv('/Users/daniel/Documents/projects/real-estate-dataset.csv')
drop_cols = ['ZN', 'INDUS', 'CHAS', 'NOX', 'DIS', 'TAX']
data = data.drop(drop_cols, axis=1)
data['B'] = ((data['B'] + 0.63) / 1000)**0.5
data = data.dropna()
values = data.values 

# describe data
print(data.head())
print(data.describe())
print(data.corr())

# plot distibution of a few 

sns.pairplot(data, kind='reg', height=1)
plt.show()

data['LSTAT'].plot(kind='hist')
plt.show()
data['MEDV'].plot(kind='hist')
plt.show()

data.plot(kind='scatter', x='LSTAT', y='MEDV')

# line best fit
LSTAT_fit_linear = np.polyfit(data['LSTAT'], data['MEDV'], 1)
print(LSTAT_fit_linear)
LSTAT_fit_exp = np.polyfit(data['LSTAT'], data['MEDV'], 2)
print(LSTAT_fit_exp)

# add lines to scatter plot
plt.plot(data['LSTAT']
        , LSTAT_fit_linear[1] + data['LSTAT'] * LSTAT_fit_linear[0]
        , color='blue', linewidth=2)
plt.text(50, 50
        ,'{:.2f} + {:.2f}*x'.format(LSTAT_fit_linear[1], LSTAT_fit_linear[0])
        , color='blue', size=12)

p = np.poly1d(LSTAT_fit_exp)
t = np.linspace(0, 50, 1024)
plt.plot(t, p(t), color='red')
plt.text(50, 75
        , '{:.2f} + {:.2f}*x + {:.2f}*x^2'.format(LSTAT_fit_exp[2], LSTAT_fit_exp[1], LSTAT_fit_exp[0])
        , color='red', size=12)

plt.show()

# simple linear regression 
lstat_fit = LinearRegression()
lstat_fit.fit(data[['LSTAT']], data['MEDV'])
print(lstat_fit.intercept_, lstat_fit.coef_)
# print R and p-value
pearson_coef, p_val = scs.pearsonr(data['LSTAT'], data['MEDV'])
print(pearson_coef, p_val)

# multilinear regression
mlr = LinearRegression()
mlr.fit(data[['LSTAT', 'CRIM', 'RAD', 'B']], data['MEDV'])
print(mlr.intercept_, mlr.coef_)
print("R-sq:", mlr.score(data[['LSTAT', 'CRIM', 'RAD', 'B']], data['MEDV']))


# simple linear regression using statsmodels (cannot contain missing values)
import statsmodels.api as sm

X = data["RM"]      # input/independent variable
Y = data["MEDV"]    # output/dependent variable
X = sm.add_constant(X)  # add an intercept to the model

model = sm.OLS(Y, X).fit()  # the order is Y and then X
preds = model.predict(X)
print(model.summary())

# multi linear regression using statsmodels

X = data[["RM", "LSTAT"]]
Y = data["MEDV"]

mlr = sm.OLS(Y, X).fit()    # the order is Y and then X
preds = mlr.predict(X)
print(mlr.summary())
