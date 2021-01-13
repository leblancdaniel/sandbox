timport warnings
import itertools 
import pandas as pd 
import numpy as np 
import statsmodels.api as sm
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

#load data
data = sm.datasets.co2.load_pandas()
y = data.data 
#resample timeseries into monthly buckets
y = y['co2'].resample('MS').mean()
#backfill any NaN values
y = y.fillna(y.bfill())

# show data
y.plot(figsize=(15,6))
plt.show()

#### Autoregression (AR) ####
"""
AR models the next step in the sequence as a linear function of observations at the prior time step
AR(1) is a first-order AR model.
AR method is best for univariate time series WITHOUT TREND and SEASONAL COMPONENTS
"""
from statsmodels.tsa.ar_model import AutoReg
ar_model = AutoReg(y, lags=1)
ar_model = ar_model.fit()
ar_yhat = ar_model.predict(len(y), len(y))
print(ar_yhat)


#### Moving Average (MA) ####
"""
MA models next step as a linear function of residual errors from a mean process at prior time steps
MA(0) is a zeroth-order MA model
MA method is best for univariate time series WITHOUT TREND AND SEASONAL COMPONENTS
"""
from statsmodels.tsa.arima_model import ARMA 
ma_model = ARMA(y, order=(0, 1))
ma_model = ma_model.fit(disp=False)
ma_yhat = ma_model.predict(len(y), len(y))
print(ma_yhat)


#### Autoregressive Moving Average (ARMA) ####
"""
ARMA method models linear function of the observations and residual errors at prior time steps
combines both AR and MA models
ARMA method is suitable for univariate time series WITHOUT TREND AND SEASONAL COMPONENTS
"""
from statsmodels.tsa.arima_model import ARMA 
arma_model = ARMA(y, order=(2, 1))
arma_model = arma_model.fit(disp=False)
arma_yhat = arma_model.predict(len(y), len(y))
print(arma_yhat)


#### Autoregressive Integrated Moving Average (ARIMA) ####
"""
ARIMA models next step as linear function of differenced observations and residual errors at prior time steps
ARIMA combines AR and MA, plus a differencing pre-processing step of sequence to make it stationary (called Integration)
ARIMA model can also be used to develop AR, MA, and ARMA models
ARIMA is suitable for univariate time series WITH TREND and WITHOUT SEASONAL COMPONENTS
"""
from statsmodels.tsa.arima_model import ARIMA 
arima_model = ARIMA(y, order=(1,1,1))
arima_model = arima_model.fit(disp=False)
arima_yhat = arima_model.predict(len(y), len(y), typ='levels')
print(arima_yhat)


#### Seasonal ARIMA (SARIMA) ####
"""
Same as ARIMA, except models linear function of differenced seasonal observations as well
Specify the order for the AR(p), I(d) and MA(q) of ARIMA, and AR(P), I(D), MA(Q) and M parameters at seasonal level
where M is the number of time steps in each season
SARIMA is suitable for univariate time series WITH TREND AND/OR SEASONAL COMPONENTS
"""
from statsmodels.tsa.statespace.sarimax import SARIMAX

# create all combinations of order in ARIMA model
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

# fit for AIC score
warnings.filterwarnings("ignore")
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            sarima_model = SARIMAX(y, 
                                    order=param, 
                                    seasonal_order=param_seasonal,
                                    enforce_stationarity=False, 
                                    enforce_invertability=False)
            results = sarima_model.fit()
            print("ARIMA{}x{} - AIC:{:.2f}".format(param, param_seasonal, results.aic))
        except:
            continue

# plug in results with lowest AIC score
sarima_model = SARIMAX(y, order=(1,1,1), seasonal_order=(0,1,1,12))
sarima_model = sarima_model.fit(disp=False)

# summary table of SARIMA
print("SARIMA summary table:")
print(sarima_model.summary().tables[1])

# show plot diagnostics
sarima_model.plot_diagnostics(figsize=(15,12))
plt.show()

# Show predictions using one-step forecast
pred = sarima_model.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=False)
pred_ci = pred.conf_int()

ax = y['1990':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One step ahead forecast', alpha=0.7)
ax.fill_between(pred_ci.index, 
                pred_ci.iloc[:, 0], 
                pred_ci.iloc[:, 1], color='k', alpha=0.2)
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()
plt.show()

# Show predictions using dynamic forecast
pred_dynamic = sarima_model.get_prediction(start=pd.to_datetime('1998-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

ax = y['1990':].plot(label='observed', figsize=(15,12))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)
ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=0.25)
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1998-01-01'), y.index[-1], alpha=0.1, zorder=-1)
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()
plt.show()

# Visualize forecasts 500 steps in the future
pred_uc = sarima_model.get_forecast(steps=500)
pred_ci = pred_uc.conf_int() 

ax = y.plot(label='observed', figsize=(15,12))
pred_uc.predicted_mean.plot(ax=ax, label='forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:,0],
                pred_ci.iloc[:,1], color='k', alpha=0.25)
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()
plt.show()


#### Seasonal ARIMA w/ Exogenous Regressors (SARIMAX) ####
"""
Extension of SARIMA that includes modeling of exogenous variables (covariates).
Exogenous variables can be thought of as parallel input sequences that have observations 
at the same time steps as the original series.
SARIMAX is suitable for univariate time series WITH TREND AND/OR SEASONAL COMPONENTS AND EXOGENOUS VARIABLES
"""
"""
from statsmodels.tsa.statespace.sarimax import SARIMAX
from random import random
data_exog = [x + random() for x in range(len(y))]
sarimax_model = SARIMAX(y, exog=data_exog, order=(1,1,1), seasonal_order=(0,0,0,0))
sarimax_model = sarimax_model.fit(disp=False)
#make prediction
exog2 = [len(y) + random()]
sarimax_yhat = sarimax_model.predict(len(y), len(y), exog=[exog2])
print(sarimax_yhat)
"""

#### Vector Autoregression (VAR) ####
"""
VAR method models next step using an AR model.  It's a generalization of AR to multivariate time series
VAR is suitable for MULTIVARIATE time series WITHOUT TREND AND SEASONAL COMPONENTS
"""
"""
from statsmodels.tsa.vector_ar.var_model import VAR
var_model = VAR(y)
var_model = var_model.fit()
# use .forecast instead of .predict for Vector models
var_yhat = var_model.forecast(var_model.y, steps=1)
"""

#### Vector ARMA (VARMA) ####
"""
VARMA models each next step using an ARMA model
VARMA is suitable for MULTIVARIATE time series WITHOUT TREND AND SEASONAL COMPONENTS
"""
"""
from statsmodels.tsa.statespace.varmax import VARMAX 
varma_model = VARMAX(y, order=(1,1))
varma_model = varma_model.fit(disp=False)
varma_yhat = varma_model.forecast()
"""

#### VARMA w/ Exogenous Regressors (VARMAX) ####
"""
Extention of VARMA to include modeling of exogenous variables
VARMAX is suitable for MULTIVARIATE time series WITHOUT TREND AND SEASONAL COMPONENTS and WITH EXOGENOUS VARIABLES
"""
"""
from statsmodels.tsa.statespace.varmax import VARMAX 
varmax_model = VARMAX(y, exog=exog_data, order=(1,1))
varmax_model = varmax_model.fit(disp=False)
exog2 = [[100]]
varmax_yhat = varmax_model.forecast(exog=exog2)
"""

#### Simple Exponential Smoothing (SES) ####
"""
SES models the next time step as exponentially weighted linear function of prior observations
SES is suitable for UNIVARIATE time series WITHOUT TREND AND SEASONAL COMPONENTS
"""
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

ses_model = SimpleExpSmoothing(y)
ses_model = ses_model.fit()
ses_yhat = ses_model.predict(pd.to_datetime('1998-01-01'), y.index[-1])
# plot results
ax = y['1990':].plot(label='observed', figsize=(15,12))
ses_yhat.plot(label='Simple Exponential Smoothing', ax=ax)
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('1998-01-01'), y.index[-1], alpha=0.1, zorder=-1)


#### Holt Winter's Exponential Smoothins (HWES) ####
"""
HWES (AKA, Triple Exponential Smoothing) models next step as exponentially weighted linear function
of prior observations, taking trend + seasonality into account.
HWES is suitable for UNIVARIATE time series WITH TREND AND/OR SEASONAL COMPONENTS
"""
from statsmodels.tsa.holtwinters import ExponentialSmoothing

hwes_model = ExponentialSmoothing(y)
hwes_model = hwes_model.fit()
hwes_yhat = hwes_model.predict(pd.to_datetime('1998-01-01'), y.index[-1])
# plot results
hwes_yhat.plot(label='Holt Winter Exponential Smooting', ax=ax, alpha=0.5)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()
plt.show()

