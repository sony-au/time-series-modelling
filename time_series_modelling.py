# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 11:37:52 2020

@author: group 23
"""

import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt

# Load the data and change the 'Months' to datetime format and set as the index
dateparse = lambda dates: pd.datetime.strptime(dates, '%b-%y')
ts = pd.read_csv('UnemploymentRateJan1986-Dec2018.csv',
                       parse_dates=['Months'],
                       index_col='Months',
                       date_parser=dateparse)   
ts = ts['Unemployment_Rates']

# log transformation of ts
ts_log = np.log(ts)

# Plot the original and log-transformed data
plt.figure(figsize=(15,6))
plt.plot(ts[:], label='original')
plt.plot(ts_log[:], color='red', label='log-transformed')
plt.title('Unemployment Rates (January 1986 - December 2018)', size=18)
plt.legend(loc='upper right', fontsize=18)
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.tick_params(labelsize=13)
plt.savefig('Unemployment Rates, Jan 1986-Dec 2018 (original vs log-transformed).png', 
            bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.show()

# Plot the seasonality (M=12, ie repeat every 12 months) of the transformed data
plt.figure(figsize=(15,6))
plt.plot(ts_log[:], color='red', label='log-transformed')
plt.title('Unemployment Rates (January 1986 - December 2018)', size=18)
plt.legend(loc='upper right', fontsize=18)
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.tick_params(labelsize=13)
plt.savefig('Unemployment Rates, Jan 1986-Dec 2018.png', 
            bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.show()

plt.figure(figsize=(15,6))
plt.plot(ts_log[-60:], color='red', label='log-transformed')
plt.title('Unemployment Rates (January 2014 - December 2018)', size=18)
plt.legend(loc='upper right', fontsize=18)
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.tick_params(labelsize=13)
plt.savefig('Unemployment Rates, Jan 2014-Dec 2018.png', 
            bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.show()

# Function to calculate MSE
def mse(x, y):
    return np.mean(np.power(x - y,2))

# Create x-axis for the plots
import datetime
start = datetime.datetime(1986, 1, 1)
end = datetime.datetime(2020, 1, 1)
x = pd.date_range(start, end, freq='M')

# =============================================================================
# PART 1: Exploratory Data Analysis using Additive decomposition method
# =============================================================================
# Estimate the trend component using CMA-12
Trend = ts_log.rolling(12, center = True).mean().rolling(2,center = True).mean().shift(-1)

# De-trend the data to obtain the estimate of the seasonal component 
ts_detrend = ts_log - Trend

plt.figure(figsize=(15,6))
plt.plot(ts_detrend, color='red', label='Seasonal Component')
plt.title('Unemployment Rates (January 1986 - December 2018)', size=18) 
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.legend(loc='upper right', fontsize=18)
plt.tick_params(labelsize=13)
plt.savefig('Unemployment Rates_Seasonal Component, Jan 1986-Dec 2018.png', 
            bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.show()

# Replace nan to 0
ts_zero = np.nan_to_num(ts_detrend)

# Check stationarity
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    # Perform Augmented Dickey-Fuller test
    print('Results of Augmented Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value',
                         '#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

def plot_curve(timeseries): 
    # Determine rolling statistics
    rolmean = timeseries.rolling(12,center=True).mean()
    rolstd = timeseries.rolling(12,center=True).std()
    #Plot rolling statistics:
    plt.figure(figsize=(15,6))
    plt.plot(timeseries, color='blue', label='time series data')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.title('Rolling Mean & Standard Deviation', size=18)
    plt.legend(loc='upper right', fontsize=13)
    plt.xlabel('Time', size=16) 
    plt.ylabel('Unemployment Rate (%)', size=16)
    plt.tick_params(labelsize=13)
    plt.savefig('Rolling Mean & Standard Deviation', 
            bbox_inches='tight', dpi=300, pad_inches=0.0)
    plt.show()

test_stationarity(ts_zero)
plot_curve(ts_detrend)

# Calculate the seasonal index
monthly_S = ts_zero.reshape(-1,12) 

# Calculate the monthly average of seasonal index
monthly_avg = np.mean(monthly_S[1:-1,:], axis=0)

# Normalise the seasonal index
mean_allmonth = monthly_avg.mean()
monthly_avg_norm = monthly_avg - mean_allmonth

# Repeat seasonal index for 33 years
tiled_avg = np.tile(monthly_avg_norm, 33)

# Calculate seasonally adjusted data
seasonal_adjusted = ts_log - tiled_avg

# Plot seasonally adjusted data
plt.figure(figsize=(15,6))
plt.plot(seasonal_adjusted, color='red',label='Seasonally Adjusted')
plt.title('Unemployment Rates (January 1986 - December 2018)', size=18) 
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.legend(loc='upper right', fontsize=18)
plt.tick_params(labelsize=13)
plt.savefig('Unemployment Rates_Seasonally Adjusted, Jan 1986-Dec 2018.png', 
            bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.show()

# Smooth the Seasonal adjusted data using CMA-12 to obtain the final trend estimate
Trend_final = seasonal_adjusted.rolling(12, 
          center = True).mean().rolling(2, 
          center = True).mean().shift(-1)

# Check the residual
residual = ts_log - Trend_final - tiled_avg

# Plot the residual
plt.figure(figsize=(15,6))
plt.plot(residual, label='Estimated Residuals')
plt.title('Unemployment Rates (January 1986 - December 2018)', size=18)
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.legend(loc='upper right', fontsize=18)
plt.tick_params(labelsize=13)
plt.savefig('Unemployment Rates_Estimated Residuals, Jan 1986-Dec 2018.png', 
            bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.show() # Some patterns can still be identified

# Estimate the cycle component
Cycle_final = residual.rolling(6, 
          center = True).mean().rolling(2, 
          center = True).mean().shift(-1) 

# Calculate the residual again
residual2 = ts_log - Trend_final - Cycle_final - tiled_avg

# Plot the residual again
plt.figure(figsize=(15,6))
plt.plot(residual,label='Estimated Residuals',color="grey",alpha=0.5) # Some patterns can still be identified
plt.plot(residual2,label='Final Residuals',color="red",alpha=0.6) # much better now
plt.title('Unemployment Rates (January 1986 - December 2018)', size=18)
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.legend(loc='upper right', fontsize=18)
plt.tick_params(labelsize=13)
plt.savefig('Unemployment Rates_Final Residuals vs Estimated Residuals, Jan 1986-Dec 2018.png', 
            bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.show()

# Histogram of residual
residual2.plot.hist(grid=False, bins=20, rwidth=0.9)
plt.title('Histogram of Residuals')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Count')
plt.savefig('Histogram of Residuals.png', 
            bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.show()

# ACF
from pandas.plotting import autocorrelation_plot
plt.figure(figsize=(15,6))
autocorrelation_plot(residual2)
plt.title('Autocorrelation')
plt.savefig('ACF (Final Residuals)', 
            bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.show()

# Plot the data
fig, ax = plt.subplots(5, 1,figsize=(12,12))
ax[0].plot(ts_log, color='red')
ax[1].plot(Trend_final)
ax[2].plot(Cycle_final)
ax[3].plot(tiled_avg)
ax[4].plot(residual2)
ax[0].legend(['log-transformed'], loc=1)
ax[1].legend(['Trend'], loc=1)
ax[2].legend(['Cycle'], loc=1)
ax[3].legend(['Seasonality'], loc=1)
ax[4].legend(['Remainder'], loc=1)
ax[4].set_xlabel('Year', size=12) 
plt.savefig('Unemployment Rates and Its Four Additive Decomposition Components.png', 
            bbox_inches='tight', dpi=300, pad_inches=0.0)
plt.show()

# =============================================================================
# PART 2: Using additive decomposition and linear regression for forecasting
# =============================================================================
# Estimate the trend component using CMA-12
Trend2 = ts_log.rolling(12, center = True).mean().rolling(2,center= True).mean().shift(-1)

# De-trend the data to obtain the estimate of the seasonal component 
ts_detrend2 = ts_log - Trend2

# Replace nan to 0
ts_zero2 = np.nan_to_num(ts_detrend2)

# Calculate the seasonal index
monthly_S2 = ts_zero2.reshape(-1,12) 

# Calculate the monthly average of seasonal index
monthly_avg2 = np.mean(monthly_S2[1:-1,:], axis=0)

# Normalise the seasonal index
mean_allmonth2 = monthly_avg2.mean()
monthly_avg_norm2 = monthly_avg2 - mean_allmonth2

# Repeat seasonal index for 33 years
tiled_avg2 = np.tile(monthly_avg_norm2, 33)

# Calculate Seasonal adjusted data
seasonal_adjusted2 = ts_log - tiled_avg2

# Re-estimate trend by using linear regression
X2 = np.linspace(0, len(ts_log)-1+12, len(ts_log)+12).reshape(-1,1)
y2 = seasonal_adjusted2.values.reshape(-1,1).reshape(-1,1)

X_train2 = X2[:-24]
X_test2 = X2[-24:-12]
X_20192 = X2[-12:]
y_train2 = y2[:-12]
y_test2 = y2[-12:]

# Train a linear regression to estimate the trend
from sklearn.linear_model import LinearRegression
lr2 = LinearRegression().fit(X_train2, y_train2)
trend_test2 = lr2.predict(X_test2)
trend_train2 = lr2.predict(X_train2)
trend_20192 = lr2.predict(X_20192)
trend_all2 = lr2.predict(X2[:-12])
print("Coefficients: {0}".format(lr2.coef_))
print("Intercept: {0}".format(lr2.intercept_))
print("Total model: y = {0} + {1} X".format(lr2.intercept_, lr2.coef_[0]))

# Final forecast
forecast_test2 = np.exp(trend_test2.reshape(-1,) + monthly_avg_norm2)
forecast_20192 = np.exp(trend_20192.reshape(-1,) + monthly_avg_norm2)
forecast_all2 = np.exp(trend_all2.reshape(-1,) + tiled_avg2)

# Plot the data and forecasts
plt.figure(figsize=(15,6))
plt.plot(x[:-12],ts,label='actual')
plt.plot(x[-24:-12],forecast_test2,label='forecast 2018')
plt.plot(x[-12:],forecast_20192,label='forecast 2019')
plt.plot(x[:-12],forecast_all2,label='forecast 2001-2018',alpha=0.7,color='grey')
plt.title('Unemployment Rates - Forecast 2019 (Linear Regression - Additive)', size=18)
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.tick_params(labelsize=13)
plt.legend(loc='upper right', fontsize=14)
plt.savefig('Unemployment Rates - Forecast 2019 (Linear Regression - Additive)',
            bbox_inches='tight', dpi=300)
plt.show()

# Calculate the mse
mse_test2 = mse(forecast_test2,ts[-12:,])
print("MSE linear regression (additive):", mse_test2)


# =============================================================================
# PART 3: Using multiplicative decomposition and linear regression for forecasting
# =============================================================================
# Estimate the trend component using CMA-12
Trend3 = ts_log.rolling(12, center = True).mean().rolling(2,center= True).mean().shift(-1)

# De-trend the data to obtain the estimate of the seasonal component 
ts_detrend3 = ts_log/Trend3

# Replace nan to 0
ts_zero3 = np.nan_to_num(ts_detrend3)

# Calculate the seasonal index
monthly_S3 = ts_zero3.reshape(-1,12) 

# Calculate the monthly average of seasonal index
monthly_avg3 = np.mean(monthly_S3[1:-1,:], axis=0)

# Normalise the seasonal index
c3 = 12/sum(monthly_avg3)
monthly_avg_norm3 = monthly_avg3*c3

# Repeat seasonal index for 33 years
tiled_avg3 = np.tile(monthly_avg_norm3, 33)

# Calculate Seasonal adjusted data
seasonal_adjusted3 = ts_log/tiled_avg3

# Plot the data
fig, ax = plt.subplots(4, 1, figsize=(12,12))
ax[0].plot(ts_log, color='red')
ax[1].plot(Trend3)
ax[2].plot(tiled_avg3)
ax[3].plot(seasonal_adjusted3)
ax[0].legend(['log-transformed'], loc=1)
ax[1].legend(['Trend-Cycle'], loc=1)
ax[2].legend(['Seasonality'], loc=1)
ax[3].legend(['seasonal adjusted'], loc=1)
ax[3].set_xlabel('Year', size=12) 
plt.savefig('Multiplicatively Decomposed Unemployment Rates.png',
            bbox_inches='tight', dpi=300)
plt.show()

# Re-estimate trend by using linear regression
X3 = np.linspace(0, len(ts_log)-1+12, len(ts_log)+12).reshape(-1,1)
y3 = seasonal_adjusted3.values.reshape(-1,1).reshape(-1,1)

X_train3 = X3[:-24]
X_test3 = X3[-24:-12]
X_20193 = X3[-12:]
y_train3 = y3[:-12]
y_test3 = y3[-12:]

# Train a linear regression to estimate the trend
from sklearn.linear_model import LinearRegression
lr3 = LinearRegression().fit(X_train3, y_train3)
trend_test3 = lr3.predict(X_test3)
trend_train3 = lr3.predict(X_train3)
trend_20193 = lr3.predict(X_20193)
trend_all3 = lr3.predict(X3[:-12])
print("Coefficients: {0}".format(lr3.coef_))
print("Intercept: {0}".format(lr3.intercept_))
print("Total model: y = {0} + {1} X".format(lr3.intercept_, lr3.coef_[0]))

# Final forecast
forecast_test3 = np.exp(trend_test3.reshape(-1,) * monthly_avg_norm3)
forecast_20193 = np.exp(trend_20193.reshape(-1,) * monthly_avg_norm3)
forecast_all3 = np.exp(trend_all3.reshape(-1,) * tiled_avg3)

# Plot the data and forecasts
plt.figure(figsize=(15,6))
plt.plot(x[:-12],ts,label='actual')
plt.plot(x[-24:-12],forecast_test3,label='forecast 2018')
plt.plot(x[-12:],forecast_20193,label='forecast 2019')
plt.plot(x[:-12],forecast_all3,label='forecast 2001-2018',alpha=0.7,color='grey')
plt.title('Unemployment Rates - Forecast 2019 (Linear Regression - Multiplicative)', size=18)
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.tick_params(labelsize=13)
plt.legend(loc='upper right', fontsize=14)
plt.savefig('Unemployment Rates - Forecast 2019 (Linear Regression - Multiplicative)',
            bbox_inches='tight', dpi=300)
plt.show()

# Calculate the mse
mse_test3 = mse(forecast_test3,ts[-12:,])
print("MSE linear regression (multiplicative):", mse_test3)


# =============================================================================
# PART 4: Additive decomposition and polynomial regression for forecasting
# =============================================================================
# Estimate the trend component using CMA-12
Trend4 = ts_log.rolling(12, center = True).mean().rolling(2,center= True).mean().shift(-1)

# De-trend the data to obtain the estimate of the seasonal component 
ts_detrend4 = ts_log - Trend4

# Replace nan to 0
ts_zero4 = np.nan_to_num(ts_detrend4)

# Calculate the seasonal index
monthly_S4 = ts_zero4.reshape(-1,12) 

# Calculate the monthly average of seasonal index
monthly_avg4 = np.mean(monthly_S4[1:-1,:], axis=0)

# Normalise the seasonal index
mean_allmonth4 = monthly_avg4.mean()
monthly_avg_norm4 = monthly_avg4 - mean_allmonth4

# Repeat seasonal index for 33 years
tiled_avg4 = np.tile(monthly_avg_norm4, 33)

# Calculate Seasonal adjusted data
seasonal_adjusted4 = ts_log - tiled_avg4

# Re-estimate trend by using linear regression
X4 = np.linspace(0, len(ts_log)-1+12, len(ts_log)+12).reshape(-1,1)
y4 = seasonal_adjusted4.values.reshape(-1,1).reshape(-1,1)

X_train4 = X4[:-36]
X_valid4 = X4[-36:-24]
X_test4 = X4[-24:-12]
X_20194 = X4[-12:]
y_train4 = y4[:-24]
y_valid4 = y4[-24:-12]
y_test4 = y4[-12:]

# Train a polynomial regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

mse_min4 = 999
for n in range(2,101):
    polynomial_features = PolynomialFeatures(degree=n)
    X_train_poly = polynomial_features.fit_transform(X_train4)
    X_valid_poly = polynomial_features.fit_transform(X_valid4)

    lr_poly = LinearRegression().fit(X_train_poly, y_train4)
    trend_valid4 = lr_poly.predict(X_valid_poly)

    # Final forecast
    forecast_valid4 = np.exp(trend_valid4.reshape(-1,) + monthly_avg_norm4)
    
    mse_valid4 = mse(forecast_valid4,ts[-24:-12])
    if mse_valid4 < mse_min4:
        mse_min4 = mse_valid4
        degree_min4 = n

print(f"Degree: {degree_min4}, min MSE: {mse_min4}")

polynomial_features = PolynomialFeatures(degree=degree_min4)
X_train_poly = polynomial_features.fit_transform(X_train4)
X_valid_poly = polynomial_features.fit_transform(X_valid4)
X_test_poly = polynomial_features.fit_transform(X_test4)
X_2019_poly = polynomial_features.fit_transform(X_20194)
X_poly = polynomial_features.fit_transform(X4)

lr_poly = LinearRegression().fit(X_train_poly, y_train4)
trend_test4 = lr_poly.predict(X_test_poly)
trend_valid4 = lr_poly.predict(X_valid_poly)
trend_train4 = lr_poly.predict(X_train_poly)
trend_20194 = lr_poly.predict(X_2019_poly)
trend_all4 = lr_poly.predict(X_poly[:-12])
print("Coefficients: {0}".format(lr_poly.coef_))
print("Intercept: {0}".format(lr_poly.intercept_))

# Final forecast
forecast_valid4 = np.exp(trend_valid4.reshape(-1,) + monthly_avg_norm4)
forecast_test4 = np.exp(trend_test4.reshape(-1,) + monthly_avg_norm4)
forecast_20194 = np.exp(trend_20194.reshape(-1,) + monthly_avg_norm4)
forecast_all4 = np.exp(trend_all4.reshape(-1,) + tiled_avg4)

# plot the data
plt.figure(figsize=(15,6))
plt.plot(x[:-12],ts,label='actual')
plt.plot(x[-36:-24],forecast_valid4,label='forecast 2017')
plt.plot(x[-24:-12],forecast_test4,label='forecast 2018')
plt.plot(x[-12:],forecast_20194,label='forecast 2019')
plt.plot(x[:-12],forecast_all4,label='forecast 2001-2018',alpha=0.7,color='grey')
plt.title('Unemployment Rates - Forecast 2019 (Polynomial Regression - Additive)', size=18)
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.tick_params(labelsize=13)
plt.legend(loc='upper right', fontsize=14)
plt.savefig('Unemployment Rates - Forecast 2019 (Polynomial Regression - Additive)',
            bbox_inches='tight', dpi=300)
plt.show()

# Calculate the mse
mse_test4 = mse(forecast_test4,ts[-12:,])
print("MSE polynomial regression (d=7):", mse_test4)


# =============================================================================
# PART 5: Using Holt-Winters' method for forecasting (Additive and Multiplicative)
# =============================================================================
# Set seasonal as additive or multiplicative
from statsmodels.tsa.holtwinters import ExponentialSmoothing

fit_add = ExponentialSmoothing(ts, seasonal_periods=12, trend='add', seasonal='add').fit()
fit_mul = ExponentialSmoothing(ts, seasonal_periods=12, trend='add', seasonal='mul').fit()

# The symbol r $ and \ in the results variable are the latex symbols for
# visualization in Jupyter notebook
results = pd.DataFrame(index=[r"$\alpha$",                            r"$\beta$",
                              r"$\phi$",                            r"$\gamma$",
                              r"$l_0$",                            "$b_0$",
                              "SSE"])

# ExponentialSmoothing() object has following attributes
params = ['smoothing_level',           'smoothing_slope',           'damping_slope',
           'smoothing_seasonal',           'initial_level',           'initial_slope']

# Check out the performance of additive and multiplicative
results["Additive"] = [fit_add.params[p] for p in params] + [fit_add.sse]
results["Multiplicative"] = [fit_mul.params[p] for p in params] + [fit_mul.sse] 

print("Unemployment rate using Holt-Winters method with \n \
      both additive and multiplicative seasonality.")
print(results)

# Display of Holt-Winters Additive and Multiplicative graph over the original one
plt.figure()
ax = ts.plot(figsize=(15,6), color='black', label='original')
ax.set_title("Forecasts from Holt-Winters' Methods", fontsize=18)
ax.set_xlabel('Time', fontsize=16)
ax.set_ylabel('Unemployment Rate (%)', fontsize=16)

# Transfer the datatype to values 
smooth_add = fit_add.fittedvalues
smooth_mul = fit_mul.fittedvalues
smooth_add.plot(ax=ax, style='-', color='red', label='additive')
smooth_mul.plot(ax=ax, style='--', color='green', label='multiplicative')
plt.tick_params(labelsize=13)
plt.legend(loc='upper right', fontsize=14)
plt.savefig("Holt-Winters' Additive and Multiplicative Graphs Over the Original Time Series",
            bbox_inches='tight', dpi=300)
plt.show()

# Spliting the test and train data. Train us upto 2017 and after TEST is from 2017- 2018
X_train5 = ts[:'2017-12-01']
X_test5 = ts['2018-01-01':]

# Forecasting using Additive Holt-Winters
fit1 = ExponentialSmoothing(np.asarray(X_train5) ,seasonal_periods=12, trend='add', 
                            seasonal='add',).fit()

# Forecast to the next 12 months
y_hat_win1 = fit1.forecast(len(X_test5))

# Plot the forecast
plt.figure(figsize=(15,6))
plt.plot(x[:-24], X_train5, label='Train' , color='red')
plt.plot(x[-24:-12], X_test5, label='Test', color='green')
plt.plot(x[-12:], y_hat_win1, label="Holt-Winters' Additive", color='orange')
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.title("Unemployment Rates - Forecast 2019 (Holt-Winters' Additive)", size=18)
plt.tick_params(labelsize=13)
plt.legend(loc='upper right', fontsize=14)
plt.savefig("Unemployment Rates - Forecast 2019 (Holt-Winters' Additive)",
            bbox_inches='tight', dpi=300)
plt.show()

# MSE for the Additive Holt-Winters Model
mse_test5 = mse(X_test5,y_hat_win1)
print("MSE Holt-Winters (Additive):",mse_test5)

# Forecasting using Multiplicative Holt-Winters
fit2 = ExponentialSmoothing(np.asarray(X_train5), seasonal_periods=12, 
                            trend='add', seasonal='mul',).fit()

# Forecast to the next 12 months
y_hat_win = fit2.forecast(len(X_test5))

# Plot the forecast
plt.figure(figsize=(15,6))
plt.plot(x[:-24], X_train5, label='Train' , color='red')
plt.plot(x[-24:-12], X_test5, label='Test', color='green')
plt.plot(x[-12:], y_hat_win, label="Holt-Winters' Multiplicative", color='orange')
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.title("Unemployment Rates - Forecast 2019 (Holt-Winters' Multiplicative)", size=18)
plt.tick_params(labelsize=13)
plt.legend(loc='upper right', fontsize=14)
plt.savefig("Unemployment Rates - Forecast 2019 (Holt-Winters' Multiplicative)",
            bbox_inches='tight', dpi=300)
plt.show()

# MSE for the Multiplicative Holt-Winters Model
mse_test6 = mse(X_test5,y_hat_win)
print("MSE Holt-Winters (multiplicative):",mse_test6)

# Compare the forecasting outcomes of additive and multiplicative
# np.c_: Translates slice objects to concatenation along the second axis.
df_add = pd.DataFrame(np.c_[ts, fit_add.level, fit_add.slope, fit_add.season, 
                      fit_add.fittedvalues],
                      columns=[r'$y_t$',r'$l_t$',r'$b_t$',r'$s_t$',r'$\hat{y}_t$'],
                      index=ts.index)
df_add = df_add.append(fit_add.forecast(24).rename(r'$\hat{y}_t$').to_frame())

df_mul = pd.DataFrame(np.c_[ts, fit_mul.level, fit_mul.slope, fit_mul.season, 
                      fit_mul.fittedvalues], 
                      columns=[r'$y_t$',r'$l_t$',r'$b_t$',r'$s_t$',r'$\hat{y}_t$'],
                      index=ts.index)
df_mul=df_mul.append(fit_mul.forecast(24).rename(r'$\hat{y}_t$').to_frame())

# Plot the level, trend and season for fit_add and fit_mul
# Define 2 states variable for convenience
states_add = pd.DataFrame(np.c_[fit_add.level, fit_add.slope, fit_add.season],
                          columns=['level','slope','seasonal'],index=ts.index)
states_mul = pd.DataFrame(np.c_[fit_mul.level, fit_mul.slope, fit_mul.season],
                          columns=['level','slope','seasonal'],index=ts.index)

# Define subplots windows
fig, [[ax1, ax4],[ax2, ax5], [ax3, ax6]] = plt.subplots(3, 2, figsize=(15,13))
states_add['level'].plot(ax=ax1)
states_add['slope'].plot(ax=ax2)
states_add['seasonal'].plot(ax=ax3)
states_mul['level'].plot(ax=ax4)
states_mul['slope'].plot(ax=ax5)
states_mul['seasonal'].plot(ax=ax6)
plt.savefig("Comparison between Holt-Winters' Additive and Multiplicative Methods",
            bbox_inches='tight', dpi=300)
plt.show()


# =============================================================================
# PART 6: Using Seasonal ARIMA method for forecasting 
# =============================================================================
# Take the difference
diff_ts = pd.Series.diff(ts)
diff_data = diff_ts.dropna()

# Take the log_diff
log_diff_ts = pd.Series.diff(ts_log)
log_diff_data = log_diff_ts.dropna()

# Plot the original dataset
plt.figure(figsize=(15,8))
plt.plot(ts, 'r-', label="Unemployment Rate")
plt.xlabel('Year')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment Rate 1986-2018')
plt.show()

# Plot the log transformed dataset
plt.figure(figsize=(15,8))
plt.plot(ts_log, 'b-', label="Unemployment Rate_Log_Transformed")
plt.xlabel('Year')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment Rate log_transformed 1986-2018')
plt.show()

# Plot the data difference transformed
plt.figure(figsize=(15,8))
plt.plot(diff_data, 'g-', label="Unemployment Rate_data_diff")
plt.xlabel('Year')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment Rate data_diff 1986-2018')
plt.show()

# Plot the data difference log transformed
plt.figure(figsize=(15,8))
plt.plot(log_diff_data, 'b-', label="Unemployment Rate_log_diff")
plt.xlabel('Year')
plt.ylabel('Unemployment Rate')
plt.title('Unemployment Rate log_diff 1986-2018')
plt.show()

# Check for stationarity 
import statsmodels as sm
import statsmodels.api as smt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Original dataset
smt.graphics.tsa.plot_acf(ts, lags=30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(ts, lags=30, alpha=0.05)
plt.show()

# log transformed dataset
smt.graphics.tsa.plot_acf(ts_log, lags=30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(ts_log, lags=30, alpha=0.05)
plt.show()

# Data difference transformed --> this is the best approach 
smt.graphics.tsa.plot_acf(diff_data, lags=50, alpha = 0.05)
smt.graphics.tsa.plot_pacf(diff_data, lags=50, alpha=0.05)
plt.show()

# Data difference log transformed
smt.graphics.tsa.plot_acf(log_diff_data, lags=30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(log_diff_data, lags=30, alpha=0.05)
plt.show()

# We can clearly see the seasonal component in the data trend at t = 12, and 24, 
# which should be plotted using SARIMA  

# Fit the model
train_ratio = 0.969697 # this will leave the last 12 months (2018 data) as the test set
split_point = int(round(len(ts)*train_ratio))

training = ts[0: split_point]
testing = ts[split_point:]
original = ts[split_point:]

model = SARIMAX(training, order=(2,1,2), seasonal_order=(1,0,0,12), 
              enforce_stationarity=False, enforce_invertibility=False)

model_fit = model.fit(disp=-1)

forecast = model_fit.forecast(len(testing))

model = SARIMAX(ts, order=(2,1,2), seasonal_order=(1,0,0,12), 
                enforce_stationarity=False, enforce_invertibility=False)

model_fit = model.fit(disp=-1)

forecast2 = model_fit.forecast(12)

print(forecast)

# Plot the model 
plt.figure(figsize=(15,6))
plt.plot(forecast, 'r', label='forecast data 2018')
plt.plot(training, 'b', label='training data')
plt.plot(forecast2, 'y', label='forecast data 2019')
plt.plot(original, 'g', label='original data')
plt.title('SARIMA forecast')
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.tick_params(labelsize=13)
plt.legend(loc='upper right', fontsize=14)

print('BIC Seasonal ARIMA:', model_fit.bic)
print('AIC Seasonal ARIMA:', model_fit.aic)
print('MSE Seasonal ARIMA:', mse(testing,forecast))

# Find the optimal parameters 
# Please DO NOT RUN THIS AS THIS WILL TAKE A LONG TIME!!! 
error=[]
BIC=[]
AIC=[]
label=[]

for p in range(0,3):
    for d in range(0,3):
        for q in range(0,3):
            for P in range(0,3):
                for D in range(0,3):
                    for Q in range(0,3):
                        model_fit=SARIMAX(training, order=(p,d,q), 
                                          seasonal_order=(P,D,Q,12), 
                                          enforce_stationarity=False, 
                                          enforce_invertibility=False).fit(disp=-1)
                        forecast=model_fit.forecast(len(testing))
                        label.append(int(str(p)+str(d)+str(q)+str(P)+str(D)+str(Q)+str(12)))
                        error.append(mse(testing,forecast))
                        AIC.append(model_fit.aic)
                        BIC.append(model_fit.bic)
                        print('ARIMA:', p, d, q, 'Seasonal:', P, D, Q)
                        del model_fit
                        del forecast

# Convert the results into a dataframe using pandas
import pandas as pd
BIC = pd.DataFrame(np.asarray(BIC).reshape(729,1) )
BIC.to_csv('BIC.csv')

AIC = pd.DataFrame(np.asarray(AIC).reshape(729,1) )
AIC.to_csv('AIC.csv')

errors = pd.DataFrame(np.asarray(error).reshape(729,1) )
errors.to_csv('errors.csv')

labels = pd.DataFrame(np.asarray(label).reshape(729,1) )
labels.to_csv('labels.csv')

# The parameter that gives the lowest MSE
model = SARIMAX(training, order=(2,0,2), seasonal_order=(1,0,0,12), 
              enforce_stationarity=False, enforce_invertibility=False)

model_fit = model.fit(disp=-1)

forecast = model_fit.forecast(len(testing))

model = SARIMAX(ts, order=(2,1,2), seasonal_order=(1,0,0,12), 
                enforce_stationarity=False, enforce_invertibility=False)

model_fit = model.fit(disp=-1)

forecast2 = model_fit.forecast(12)

print(forecast)

# Plot the model 
plt.figure(figsize=(15,6))
plt.plot(forecast, 'r', label='forecast data 2018')
plt.plot(training, 'b', label='training data')
plt.plot(forecast2, 'y', label='forecast data 2019')
plt.plot(original, 'g', label='original data')
plt.title('Unemployment Rates - Forecast 2019 (SARIMA)', size=18)
plt.xlabel('Time', size=16) 
plt.ylabel('Unemployment Rate (%)', size=16)
plt.tick_params(labelsize=13)
plt.legend(loc='upper right', fontsize=14)
plt.savefig('Unemployment Rates - Forecast 2019 (SARIMA)',
            bbox_inches='tight', dpi=300)
plt.show()

print('BIC Seasonal ARIMA:', model_fit.bic)
print('AIC Seasonal ARIMA:', model_fit.aic)
print('MSE Seasonal ARIMA:', mse(testing,forecast))

# Calculate and plot the residuals
residuals_SARIMA=pd.DataFrame(model_fit.resid)[5:]

plt.figure(figsize=(15,8))
plt.plot(pd.DataFrame(residuals_SARIMA))
plt.show()

smt.graphics.tsa.plot_acf(residuals_SARIMA, lags=30, alpha = 0.05)
smt.graphics.tsa.plot_pacf(residuals_SARIMA, lags=30, alpha=0.05)
plt.show()


# =============================================================================
# PART 7: Using Artificial Neural Network for forecasting 
# =============================================================================
# Import all necessary modules
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np

# Set the random seed so we can get reproducible results
np.random.seed(42)
tf.random.set_seed(42)

# Set the training data
ts_train = ts[:-12]

# Scale the time series
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the min-max scaler only on the train data to avoid information leakage
fitted_transformer = scaler.fit(ts_train.values.reshape(-1,1))

# Scale both the train and test data
data = fitted_transformer.transform(ts.values.reshape(-1,1))

# Define our time window
time_window = 24

# Convert the data into X with size equal to the time window
# Get y as the output for each time window
X7, Y7 = [], []

for i in range(time_window, len(data)):
    X7.append(data[i-time_window:i, 0])
    Y7.append(data[i, 0])

# Convert X and Y to numpy array
X7 = np.array(X7)
Y7 = np.array(Y7)

# Split the data into train test validation
# Train data is all data minus the test
X_train7 = X7[:-12, :]
Y_train7 = Y7[:-12]

# Validation data is the last few months from the training data
# This is only for our early stopping
X_val7 = X7[-36:-12, :]
Y_val7 = Y7[-36:-12]

# Test data is the last few months from the training data
# plus additional 12 months of unseen data
# This is actually our validation data
X_test7 = X7[-24:, :]
Y_test7 = Y7[-24:]

# Create and Train ANN Model
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# Define our ANN model
model7 = Sequential()

# Input layer
model7.add(Dense(100, input_dim=time_window, activation='relu'))

# Dropout Layer
model7.add(Dropout(0.2))

# Hidden layers
model7.add(Dense(45, activation='linear'))
model7.add(Dense(15, activation='linear'))

# Output layer
model7.add(Dense(1))

# Compile the ANN model
model7.compile(optimizer = 'adadelta', loss = 'mse')

# Please install pydot if it's not already on your computer
# pip install pydot
# from keras.utils import plot_model
# plot_model(model7, to_file='model_NN.png', show_shapes = True, show_layer_names = True)

# Print model summary
model7.summary()

# Fit the ANN model, use early stopping with patience=20 to avoid overfitting
history7 = model7.fit(X_train7, Y_train7, validation_data = (X_val7, Y_val7),
                    epochs=150, batch_size=20, verbose = 1, 
                    callbacks=[EarlyStopping(patience=20,monitor='val_loss')])

# Plot the validation loss
plt.figure(figsize=(15, 6))
plt.title('Validation Loss vs Training Loss', size=18)
plt.plot(history7.history['val_loss'], label='Validation Loss')
plt.plot(history7.history['loss'], label='Training Loss')
plt.legend(loc='upper right', fontsize=14)
plt.savefig('Validation Loss vs Training Loss',
            bbox_inches='tight', dpi=300)
plt.show()

# Get our out of sample prediction
# Define the test size
test_size7 = 12
start7 = -test_size7*2

prediction7 = np.copy(data[:len(data) -test_size7])

# Recursivelly call the ANN model and append the prediction to 
# Get multi step ahead prediction
for i in range(len(data)-test_size7, len(data)+test_size7):
    last_feature = np.reshape(prediction7[i-time_window:i], (1,time_window))
    next_pred = model7.predict(last_feature)
    prediction7 = np.append(prediction7, next_pred)

# Create x-axis for the plot
import datetime
start_date = datetime.datetime(1986, 1, 1)
end_date = datetime.datetime(2019, 12, 31)
x_date = pd.date_range(start_date, end_date, freq='M')

# Reshape and convert the scaled prediction to original form
prediction7 = prediction7.reshape(-1,1)
prediction7 = scaler.inverse_transform(prediction7)[-24:]
data_unscaled7 = ts[:-test_size7]
test_data7 = ts[-12:]

# Plot the resulting forecasts
plt.figure(figsize=(15, 6))
plt.title('Validation Prediction Result', size=18)
plt.plot(x_date[:-24],data_unscaled7, label='Training Data')
plt.plot(x_date[-24:-12], test_data7, 'g', label='Testing Data')
plt.plot(x_date[-24:], prediction7,'r', label='Out of Sample Prediction')
plt.xlabel('Time', size=16)
plt.ylabel('Unemployment Rate (%)', size=16)
plt.legend(loc = 'upper right', fontsize=14)
plt.savefig('Validation Prediction Result (ANN)',
            bbox_inches='tight', dpi=300)
plt.show()

# Calculate the test mse
print("MSE Artifical Neural Network (model 1):", mse(prediction7[-24:-12].reshape(-1,),ts[-12:]))

# Show 12 months predictions:
print('First 12-months Predictions:\n',prediction7[-12:])

# Store the results in a csv file
final_result = pd.DataFrame({"Months":x_date[-12:], 
                         "Unemployment_Rates":np.round(prediction7[-12:].reshape(-1,),3),
                           })
final_result.set_index('Months',inplace=True)
final_result.to_csv('Predicted_Unemployment_Rates.csv', date_format='%b-%Y') 

# Train the model again using the entire data to forecast 2019
# Clear session  
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

test_size = 12
start = -test_size*2
n = 150

# Build the layers
model8 = Sequential()

model8.add(Dense(100, input_dim=time_window, activation='relu'))
model8.add(Dropout(0.2))
model8.add(Dense(45, activation='linear'))
model8.add(Dense(15, activation='linear'))
model8.add(Dense(1))
model8.compile(optimizer = 'adadelta', loss = 'mse')

# Fit the model
model8.fit(X7, Y7, validation_data = (X_test7, Y_test7),epochs=150, batch_size=20, 
              verbose = 0, callbacks=[EarlyStopping(patience=20,monitor='val_loss')])

# Make the predictions    
prediction2 = np.copy(data[:len(data)])

# Recursivelly call the ANN model and append the prediction to 
# Get multi step ahead prediction
for i in range(len(data) - test_size, len(data)+test_size):
    last_feature = np.reshape(prediction2[i-time_window:i], (1,time_window))
    next_pred = model8.predict(last_feature)
    prediction2 = np.append(prediction2, next_pred)

# Reshape and convert the scaled prediction to original form        
prediction2 = prediction2.reshape(-1,1)
prediction2 = scaler.inverse_transform(prediction2)[-24:]

# Plot the predictions
plt.plot(prediction2[-24:])

test_data8 = ts[-12:]
data_unscaled8 = ts[:-test_size]

# Plot the forecasts
plt.figure(figsize=(15, 6))
plt.title('Unemployment Rates - Forecast 2019 (ANN)', size=18)
plt.plot(x_date[:-24],data_unscaled8, label='Training Data')
plt.plot(x_date[-24:-12], test_data8, 'g', label='Testing Data')
plt.plot(x_date[-24:-12], prediction2[-24:-12], 'orange', label='In Sample Prediction')
plt.plot(x_date[-12:], prediction2[-12:], 'r', label='Out of Sample Prediction')
plt.xlabel('Time', size=16)
plt.ylabel('Unemployement Rate (%)', size=16)
plt.legend(loc = "upper right", fontsize=14)
plt.savefig('Unemployment Rates - Forecast 2019 (ANN)',
            bbox_inches='tight', dpi=300)
plt.show()

# Calculate the mse
print('In sample MSE Artifical Neural Network (model 2):', mse(prediction2[-24:-12].reshape(-1,),ts[-12:]))

print('First 12-months Out of Sample Predictions:\n', prediction2[-12:])

# Store the results in a csv file
final_result = pd.DataFrame({"Months":x_date[-12:], 
                         "Predicted_Unemployment_Rates":np.round(prediction2[-12:].reshape(-1,),3),
                           })
final_result.set_index('Months',inplace=True)
final_result.to_csv('Group023_Project_Results.csv', date_format='%b-%Y')
