import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
from scipy.stats import norm

df = pd.read_csv('daily_means_mauna_loa_apr7.csv')
                 
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df.index = pd.DatetimeIndex(df['Date'])
df.set_index('Date', inplace=True)
df = df.asfreq('D')
df = df.interpolate(method='linear')

model = ExponentialSmoothing(
    df['CO2'],
    trend='add',
    seasonal='add',
    seasonal_periods=365
)

def plot_forecast(forecast,fit, lookback_days=365, confidence_interval=0.95):
    residual_std = fit.resid.std()
    z = norm.ppf(1 - (1 - confidence_interval) / 2)
    # For each forecast step h (starting at 1), the standard error is sigma * sqrt(h)
    forecast_horizon = np.arange(1, len(forecast) + 1)
    conf_int_lower = forecast - z * residual_std * np.sqrt(forecast_horizon)
    conf_int_upper = forecast + z * residual_std * np.sqrt(forecast_horizon)

    # Plotting the forecast with confidence intervals
    plt.figure(figsize=(12, 6))
    plt.fill_between(forecast.index, conf_int_lower, conf_int_upper, color='gray', alpha=0.2,
                     label=f'{int(confidence_interval*100)}% Confidence Interval')
    plt.plot(df.index[-lookback_days:], df['CO2'][-lookback_days:], label='Historical CO₂')
    plt.plot(forecast.index, forecast, label='30-Day Forecast', color='red')
    plt.axhline(y=forecast.mean(), color='green', linestyle='--', label='Forecast Mean')
    plt.xlabel('Date')
    plt.ylabel('CO₂ (ppm)')
    plt.title('CO₂ Forecast for the Next 30 Days using Holt-Winters Exponential Smoothing')
    plt.legend()
    plt.show()
    
def plot_residuals(fit):
    # Plot of residuals with a histogram
    residuals = fit.resid
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(residuals)
    plt.title('Residuals of the Model')
    plt.xlabel('Date')
    plt.ylabel('Residuals')
    plt.subplot(2, 1, 2)
    plt.hist(residuals, bins=30, density=True, alpha=0.6, color='g')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')

    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()



fit = model.fit(optimized=True)
print(fit.summary())

forecast = fit.forecast(30)

observed_april  = df[(df.index.year == 2025) & (df.index.month == 4)]['CO2']
forecast_april = forecast[forecast.index.month == 4]

last_obs_date = observed_april.index.max()  
forecast_horizons = (forecast_april.index - last_obs_date).days

overall_mean = (observed_april.sum() + forecast_april.sum()) / (len(observed_april) + len(forecast_april))
print("April 2025 CO2 Average: ", overall_mean)

def compute_mean_confidence_interval_forecast(observed, forecast, forecast_horizons, sigma, confidence=0.95):

    n_total = len(observed) + len(forecast)
    overall_mean = (np.sum(observed) + np.sum(forecast)) / n_total
    # Only forecasted values contribute uncertainty.
    # Total variance is sigma^2 times the sum of forecast horizons.
    total_variance = sigma**2 * np.sum(forecast_horizons)
    se_mean = np.sqrt(total_variance) / n_total
    z = norm.ppf(1 - (1 - confidence) / 2)
    margin = z * se_mean
    return overall_mean - margin, overall_mean + margin

residual_std = fit.resid.std()

ci_95 = compute_mean_confidence_interval_forecast(observed_april.values, 
                                                  forecast_april.values, 
                                                  forecast_horizons.values, 
                                                  residual_std, 
                                                  confidence=0.95)
ci_99 = compute_mean_confidence_interval_forecast(observed_april.values, 
                                                  forecast_april.values, 
                                                  forecast_horizons.values, 
                                                  residual_std, 
                                                  confidence=0.99)

print(f"95% Confidence Interval for April 2025 CO2 Average: {ci_95}")
print(f"99% Confidence Interval for April 2025 CO2 Average: {ci_99}")
    

# plot prediction
#plot_forecast(forecast, fit)

# Plot residuals
#plot_residuals(fit)


one_year_forecast = fit.forecast(365)
#plot_forecast(one_year_forecast, fit, lookback_days=365*3)