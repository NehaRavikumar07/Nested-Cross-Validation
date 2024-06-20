

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('train.csv', parse_dates=['date'])
print(df.head(5))

# Ensure the relevant columns are numeric and handle missing values
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
df = df.dropna(subset=['price', 'quantity'])

# Example of nested time series: Group by city
groups = df.groupby('city')

# Store models and results
models = {}
results = {}

# Loop through each group
for name, group in groups:
    print(f'Processing {name}')
    
    # Ensure the group is sorted by date
    group = group.sort_values(by='date')
    
    # Aggregating the sales data (price * quantity) by date
    group['sales'] = group['price'] * group['quantity']
    daily_sales = group.resample('D', on='date').sum()['sales']
    
    # Fit a model, e.g., Exponential Smoothing
    try:
        model = ExponentialSmoothing(daily_sales, trend='add', seasonal='add', seasonal_periods=7).fit()
        models[name] = model
        forecast = model.forecast(steps=10)
        results[name] = forecast
    except Exception as e:
        print(f"Error processing {name}: {e}")

# Ensure city 'city1' exists in the dataset
if 'city1' in models:
    print(models['city1'].summary())
    print(results['city1'])
else:
    print("City 'city1' not found in the dataset")

# Test Case 1: Check if the model works for a single city
def test_single_city(city_name):
    if city_name in df['city'].unique():
        city_data = df[df['city'] == city_name]
        city_data = city_data.sort_values(by='date')
        city_data['sales'] = city_data['price'] * city_data['quantity']
        daily_sales = city_data.resample('D', on='date').sum()['sales']
        model = ExponentialSmoothing(daily_sales, trend='add', seasonal='add', seasonal_periods=7).fit()
        forecast = model.forecast(steps=10)
        assert len(forecast) == 10
    else:
        print(f"City '{city_name}' not found in the dataset")

test_single_city('city1')

# Function to perform cross-validation
def time_series_cv(data, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errors = []

    for train_index, test_index in tscv.split(data):
        train, test = data[train_index], data[test_index]
        model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7).fit()
        forecast = model.forecast(len(test))
        error = mean_squared_error(test, forecast)
        errors.append(error)

    return np.mean(errors)

# Example usage with one city
if 'city1' in df['city'].unique():
    city_data = df[df['city'] == 'city1']
    city_data = city_data.sort_values(by='date')
    city_data['sales'] = city_data['price'] * city_data['quantity']
    daily_sales = city_data.resample('D', on='date').sum()['sales'].values
    cv_error = time_series_cv(daily_sales, n_splits=5)
    print(f'Cross-Validation Error: {cv_error}')
else:
    print("City 'city1' not found in the dataset")

# Calculate performance metrics for all cities
performance = {}
for name, group in groups:
    group = group.sort_values(by='date')
    group['sales'] = group['price'] * group['quantity']
    daily_sales = group.resample('D', on='date').sum()['sales']
    model = models.get(name)
    if model:
        forecast = results[name]
        true_values = daily_sales[-len(forecast):]  # Assuming the last 'len(forecast)' entries are the test set
        mse = mean_squared_error(true_values, forecast)
        performance[name] = mse

print(performance)

# Plotting the results for a single city
def plot_forecast(city_name):
    if city_name in groups.groups:
        group = groups.get_group(city_name).sort_values(by='date')
        group['sales'] = group['price'] * group['quantity']
        daily_sales = group.resample('D', on='date').sum()['sales']
        model = models.get(city_name)
        if model:
            forecast = results[city_name]
            plt.figure(figsize=(10, 6))
            plt.plot(daily_sales, label='True Values')
            plt.plot(forecast, label='Forecast', linestyle='--')
            plt.title(f'Forecast vs True Values for {city_name}')
            plt.legend()
            plt.show()
        else:
            print(f"No model found for {city_name}")
    else:
        print(f"City '{city_name}' not found in the dataset")

# Example plot
plot_forecast('city1')

# Summarizing the results
summary = pd.DataFrame.from_dict(performance, orient='index', columns=['MSE'])
print(summary)

daily_sales = df.groupby('date').sum()['sales']





