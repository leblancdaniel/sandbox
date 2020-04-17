# import libraries for data reading, cleansing, processing, visualizations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import describe
import researchpy as rp
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(4)

data = pd.read_csv("C:/Users/Daniel LeBlanc/Desktop/Daniel 2019/Resume/MLB/ds1.csv")
data.columns = ['ID', 'x1', 'x2', 'x3', 'x5', 'x6', 'ya', 'yb', 'yc']

data = data.drop(['ID'], axis=1)

print(data.head())
print(data.info())

# showing histograms for all features
data.hist(bins=25, color='red', edgecolor='black', linewidth=1,
            xlabelsize=8, ylabelsize=8, grid=False)
plt.tight_layout(rect=(1, 1, 1.2, 1.2))

"""
If skewness is positive, the data are skewed right, meaning that the right tail is longer.
If skewness is negative, the data are skewed left, meaning that the left tail is longer.
If skewness = 0, the data are perfectly symmetrical.

If skewness is less than −1 or greater than +1, the distribution is highly skewed.
If skewness is between −1 and −½ or between +½ and +1, the distribution is moderately skewed.
If skewness is between −½ and +½, the distribution is approximately symmetric.

Kurtosis is a measure of peakedness (or flatness) of a distribution
 If the kurtosis is close to 0, then a normal distribution is often assumed.  These are called mesokurtic distributions.
 If the kurtosis is less than zero, then the distribution is light tails and is called a platykurtic distribution.
 If the kurtosis is greater than zero, then the distribution has heavier tails and is called a leptokurtic distribution
"""
"""
# transform non-normal distributions into normal distributions for heatmap
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
data['x1_qt'] = qt.fit_transform(data['x1'].values.reshape(-1,1))
data['x2_qt'] = qt.fit_transform(data['x2'].values.reshape(-1,1))
data['x3_qt'] = qt.fit_transform(data['x3'].values.reshape(-1,1))
data['x5_qt'] = qt.fit_transform(data['x5'].values.reshape(-1,1))
data['x6_qt'] = qt.fit_transform(data['x6'].values.reshape(-1,1))
data['ya_qt'] = qt.fit_transform(data['ya'].values.reshape(-1,1))
data['yb_qt'] = qt.fit_transform(data['yb'].values.reshape(-1,1))
data['yc_qt'] = qt.fit_transform(data['yc'].values.reshape(-1,1))
qt_features = ['x1_qt', 'x2_qt', 'x3_qt', 'x5_qt', 'x6_qt', 'ya_qt', 'yb_qt', 'yc_qt']
# remove outliers (datapoints w/ Z-score of 3 relative to mean, std devation)
print("Removing outliers of non-null values...")
data = data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]
print(data.shape)
"""
data['ya_sqrt'] = np.sqrt(data['ya'])
data['x1_sq'] = data['x1']**2
data['x3_sqrt'] = np.sqrt(data['x3'])
features = ['x1', 'x1_sq', 'x2', 'x3', 'x3_sqrt', 'x5', 'x6', 'ya', 'ya_sqrt', 'yb', 'yc']
# show heatmap to identify relationships between variables
corr = data[features].corr()
sns.heatmap(round(corr,2), annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

sns.pairplot(data[features], height=1, aspect=1,
                plot_kws=dict(s=1, edgecolor='b'), diag_kind="kde")
plt.show()
"""
#Ya relationships detail
plt.scatter(data['x1_qt'], data['ya'], s=1, color='black')
plt.scatter(data['x2_qt'], data['ya'], s=1, color='blue')
plt.scatter(data['x3_qt'], data['ya'], s=1, color='red')
plt.show()
"""
#Yb relationships detail
#plt.scatter(data['x1_qt'], data['ya'], color='black')
#plt.scatter(data['x3'], data['ya'], color='red')
#plt.show()

#Yc relationships detail
#plt.scatter(data['ya_qt'], data['yc_qt'], color='black')
#plt.scatter(data['yb_qt'], data['yc_qt'], color='red')
#plt.show()

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics
# define dependent variables to predict ya
X = data[['x1', 'x2', 'x3']].values
ya = data['ya'].values
# separate dataset into 80/20 train-test split
train, test, ya_train, ya_test = train_test_split(X, ya, train_size=0.80)
# train MLR model
regressor = LinearRegression().fit(train, ya_train)
ya_pred = regressor.predict(test)
print("Ya coefficient: ", regressor.coef_)
print("Ya intercept: ", regressor.intercept_)
print("Ya MAE: ", metrics.mean_absolute_error(ya_test, ya_pred))
print("Ya RMSE: ", np.sqrt(metrics.mean_squared_error(ya_test, ya_pred)))
print("Ya R-sq: ", metrics.r2_score(ya_test, ya_pred))

"""
# define dependent variables to predict yb
X = data[['x1', 'x3']].values
yb = data['yb'].values
# separate dataset into 80/20 train-test split
train, test, yb_train, yb_test = train_test_split(X, yb, train_size=0.80)
# train MLR model
regressor = LinearRegression().fit(train, yb_train)
yb_pred = regressor.predict(test)
print("Yb coefficient: ", regressor.coef_)
print("Yb intercept: ", regressor.intercept_)
print("Yb MAE: ", metrics.mean_absolute_error(yb_test, yb_pred))
print("Yb RMSE: ", np.sqrt(metrics.mean_squared_error(yb_test, yb_pred)))
print("Yb R-sq: ", metrics.r2_score(yb_test, yb_pred))
"""
"""
# define dependent variables to predict yc
X = data[['ya', 'yb']].values
yc = data['yc'].values
# separate dataset into 80/20 train-test split
train, test, yc_train, yc_test = train_test_split(X, yc, train_size=0.80)
# train MLR model
regressor = LinearRegression().fit(train, yc_train)
yc_pred = regressor.predict(test)
print("Yc coefficient: ", regressor.coef_)
print("Yc intercept: ", regressor.intercept_)
print("Yc MAE: ", metrics.mean_absolute_error(yc_test, yc_pred))
print("Yc RMSE: ", np.sqrt(metrics.mean_squared_error(yc_test, yc_pred)))
print("Yc R-sq: ", metrics.r2_score(yc_test, yc_pred))
"""
"""
# transform non-normal distributions into normal distributions for heatmap
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
data['ya_qt'] = qt.fit_transform(data['ya'].values.reshape(-1,1))
data['yb_qt'] = qt.fit_transform(data['yb'].values.reshape(-1,1))
data['yc_qt'] = qt.fit_transform(data['yc'].values.reshape(-1,1))
# define dependent variables to predict yc(scaled)
X = data[['ya_qt', 'yb_qt']].values
yc = data['yc_qt'].values
# separate dataset into 80/20 train-test split
train, test, yc_train, yc_test = train_test_split(X, yc, train_size=0.80)
# train MLR model
regressor = LinearRegression().fit(train, yc_train)
yc_pred = regressor.predict(test)
print("Yc_qt coefficient: ", regressor.coef_)
print("Yc_qt intercept: ", regressor.intercept_)
print("Yc_qt MAE: ", metrics.mean_absolute_error(yc_test, yc_pred))
print("Yc_qt RMSE: ", np.sqrt(metrics.mean_squared_error(yc_test, yc_pred)))
print("Yc_qt R-sq: ", metrics.r2_score(yc_test, yc_pred))
"""
