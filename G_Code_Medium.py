# ## Data Preparation

# %%
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load the dataset
file_path = "D:\Coding\Energy-Management_Project\data\data_15min.csv"
df = pd.read_csv(file_path)

# %%
# Show available columns and get user input
print("Available columns:")
print(df.columns.tolist())

target_col = input("Enter the exact name of the appliance column you want to analyze: ")

# Validate if the column exists
if target_col not in df.columns:
    raise ValueError(f"Column '{target_col}' not found in the dataset.")

# %%
# Generate a short name for the column
def generate_short_name(full_name):
    """Generate a short name from the full column name"""
    short_name = full_name.replace('DE_KN_', '')
    short_name = short_name.replace('industrial', 'ind').replace('residential', 'res')
    short_name = short_name.replace('public', 'pub')
    short_name = short_name.replace('grid_import', 'imp').replace('grid_export', 'exp')
    short_name = short_name.replace('heat_pump', 'hp').replace('washing_machine', 'wm')
    short_name = short_name.replace('dishwasher', 'dw').replace('freezer', 'frz')
    short_name = short_name.replace('circulation_pump', 'circ_pump')
    short_name = short_name.replace('photovoltaic', 'pv').replace('solar', 'slr')
    short_name = short_name.replace('battery', 'bat').replace('storage', 'stor')
    short_name = short_name.replace('charge', 'ch').replace('decharge', 'dch')
    short_name = short_name.replace('machine', 'mach').replace('aggregate', 'agg')
    short_name = short_name.replace('ventilation', 'vent').replace('refrigerator', 'fridge')
    short_name = short_name.replace('compressor', 'comp').replace('cooling', 'cool')
    short_name = short_name.replace('_', '')  # Remove remaining underscores
    return short_name

short_name = generate_short_name(target_col)
print(f"Using short name: {short_name}")

# %%
# Handle timestamp column (check which one exists)
timestamp_col = None
if 'utc_timestamp' in df.columns:
    timestamp_col = 'utc_timestamp'
elif 'cet_cest_timestamp' in df.columns:
    timestamp_col = 'cet_cest_timestamp'
else:
    raise ValueError("No timestamp column found in dataset")

# Convert timestamp to datetime and set index
df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors='coerce')
df.set_index(timestamp_col, inplace=True)

# %%
# Filter only the required column
df = df[[target_col]].copy()

# %%
# Rename target column to short name
df.rename(columns={target_col: short_name}, inplace=True)

# %%
# Drop missing values
df.dropna(inplace=True)

# %%
# Check data after cleaning
print("Data after cleaning:")
print(df.head())
print("Data range:", df.index.min(), "to", df.index.max())
print("Data frequency check (first 5 diffs):\n", df.index.to_series().diff().head())

# %% [markdown]
# ### Convert cumulative data into actual per-interval usage

# %%
# Convert cumulative to actual usage
df[short_name] = df[short_name].diff()

# Drop the first row which becomes NaN after diff
df.dropna(inplace=True)

# %%
# Optional sanity check for negative values
print("Negative values count:", (df[short_name] < 0).sum())
df = df[df[short_name] >= 0]  # or df[short_name] = df[short_name].clip(lower=0)

# %% [markdown]
# ## Data Viz

# %%
# Plot the time series consumption
plt.figure(figsize=(15,5))
plt.plot(df.index, df[short_name], label='Consumption')
plt.title(f'{target_col} Electricity Consumption Over Time')
plt.xlabel('Date')
plt.ylabel('Consumption')
plt.legend()
plt.show()

# %%
# Distribution plot of consumption
plt.figure(figsize=(8,4))
sns.histplot(df[short_name], bins=50, kde=True)
plt.title('Distribution of Consumption Values')
plt.xlabel('Consumption')
plt.show()

# %%
# Rolling mean and rolling std to check trend and seasonality
rolling_window = 96  # 96 * 15min = 24 hours
plt.figure(figsize=(15,5))
plt.plot(df[short_name], label='Original')
plt.plot(df[short_name].rolling(window=rolling_window).mean(), label='24h Rolling Mean', color='orange')
plt.plot(df[short_name].rolling(window=rolling_window).std(), label='24h Rolling Std', color='green') 
plt.title('Rolling Mean & Std of Consumption')
plt.legend()
plt.show()

# %% [markdown]
# ## Time-Series Viz

# %%
plt.figure(figsize=(15,5))
df[short_name].plot()
plt.title(f"Electricity Consumption Over Time ({short_name})")
plt.ylabel("Consumption (kWh)")
plt.xlabel("Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
df_zoom = df['2016-01-01':'2016-06-30']
plt.figure(figsize=(15,5))
df_zoom[short_name].plot()
plt.title(f"Zoomed Consumption ({short_name} - Jan-June 2016)")
plt.ylabel("Consumption (kWh)")
plt.xlabel("Time")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Resample to daily average consumption
df_daily = df[short_name].resample('D').mean()

# Filter for Jan to Jun 2016
df_daily_subset = df_daily.loc['2016-01-01':'2016-06-30']

# Plot
plt.figure(figsize=(15,4))
df_daily_subset.plot()
plt.title(f"Daily Average Consumption ({short_name}) – Jan to Jun 2016")
plt.ylabel("kWh")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Adding ON/OFF Column

# %%
# Create ON/OFF column: 1 if appliance was ON, 0 if OFF
threshold = 0.04
df[f'{short_name}_onoff'] = (df[short_name] > threshold).astype(int)

# %%
# Check result
print(df[[short_name, f'{short_name}_onoff']].head(20))
print("\nValue counts:")
print(df[f'{short_name}_onoff'].value_counts())

# %%
# Count total ON and OFF states
on_off_counts = df[f'{short_name}_onoff'].value_counts().rename(index={0: 'OFF', 1: 'ON'})
print(f"{target_col} ON/OFF Summary:")
print(on_off_counts)

# %%
# ON/OFF Plot
plt.figure(figsize=(15,4))
plt.plot(df.index[:500], df[short_name][:500], label="Consumption")
plt.plot(df.index[:500], df[f'{short_name}_onoff'][:500]*df[short_name].max(), 
         label="ON/OFF signal", alpha=0.5)
plt.legend()
plt.title("Check ON/OFF Classification vs Consumption")
plt.grid(True)
plt.show()

# %%
# Check the distribution when ON
print(df[df[f'{short_name}_onoff'] == 1][short_name].describe())
print(f"\nVery low ON readings (< {threshold} kWh):", (df[short_name] < threshold).sum())

# %%
# Plot histogram of ON consumption
plt.figure(figsize=(10,4))
sns.histplot(df[df[f'{short_name}_onoff'] == 1][short_name], bins=100, kde=True)
plt.title("Distribution of ON Consumption Values")
plt.xlabel("Consumption (kWh)")
plt.grid(True)
plt.show()

# %% [markdown]
# ## Feature Creation (Date time features)

# %%
df_feat = df.copy()
df_feat['hour'] = df_feat.index.hour
df_feat['dayofweek'] = df_feat.index.dayofweek
df_feat['quarter'] = df_feat.index.quarter
df_feat['month'] = df_feat.index.month
df_feat['year'] = df_feat.index.year
df_feat['dayofyear'] = df_feat.index.dayofyear
df_feat['dayofmonth'] = df_feat.index.day
df_feat['weekofyear'] = df_feat.index.isocalendar().week.astype(int)

# %%
# Check the new dataframe structure
print(df_feat.head())

# %% [markdown]
# ## Visualize Feature-Target Relationship (for ON data only)

# %%
# Filter ON rows only
df_on = df[df[f'{short_name}_onoff'] == 1].copy()

# %%
# Create time features for df_on
df_on['hour'] = df_on.index.hour
df_on['dayofweek'] = df_on.index.dayofweek

# %%
# Distribution of consumption during each hour of the day
plt.figure(figsize=(10,5))
sns.boxplot(x='hour', y=short_name, data=df_on)
plt.title(f'Consumption by Hour ({target_col} - ON)')
plt.show()

# %%
plt.figure(figsize=(10,5))
sns.boxplot(x='dayofweek', y=short_name, data=df_on)
plt.title(f'Consumption by Day of Week ({target_col} - ON)')
plt.show()

# %%
# Group-wise statistics by hour
print(df_on.groupby('hour')[short_name].agg(['mean', 'median', 'std', 'count']))

# %%
# Summary statistics by day of week
print(df_on.groupby('dayofweek')[short_name].describe())

# %%
# Count how many times the appliance is ON in each hour
print(df[f'{short_name}_onoff'].groupby(df.index.hour).sum())

# %% [markdown]
# ## Train Test Split (time order series)

# %%
# Define the exact time range for training and testing
train_start = pd.Timestamp("2016-01-01 00:00:00+00:00")
train_end = pd.Timestamp("2016-06-30 23:45:00+00:00")
test_start = pd.Timestamp("2016-07-01 00:00:00+00:00")
test_end = pd.Timestamp("2016-07-07 23:45:00+00:00")

# %%
# Create boolean masks
train_mask = (df_feat.index >= train_start) & (df_feat.index <= train_end)
test_mask = (df_feat.index >= test_start) & (df_feat.index <= test_end)

# %%
# Select features and target columns
feature_cols = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
target_onoff_col = f'{short_name}_onoff'

# %%
# Split the feature matrix and target vector for train and test
X_train = df_feat.loc[train_mask, feature_cols].copy()
y_train = df_feat.loc[train_mask, target_onoff_col]

X_test = df_feat.loc[test_mask, feature_cols].copy()
y_test = df_feat.loc[test_mask, target_onoff_col]

# %%
# Print shapes and summary counts to verify
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train ON count: {y_train.sum()}, Test ON count: {y_test.sum()}")

# %%
# Create lag and rolling features
df_feat[f'lag_1'] = df_feat[short_name].shift(1).fillna(0)
df_feat[f'lag_2'] = df_feat[short_name].shift(2).fillna(0)
df_feat[f'rolling_mean_4'] = df_feat[short_name].rolling(window=4).mean().fillna(0)
df_feat[f'rolling_mean_12'] = df_feat[short_name].rolling(window=12).mean().fillna(0)
df_feat[f'rolling_std_4'] = df_feat[short_name].rolling(window=4).std().fillna(0)
df_feat[f'rolling_std_12'] = df_feat[short_name].rolling(window=12).std().fillna(0)

# %%
# Update feature columns list with new features
extended_features = feature_cols + [f'lag_1', f'lag_2', f'rolling_mean_4', 
                                   f'rolling_mean_12', f'rolling_std_4', f'rolling_std_12']

# %%
# Prepare final train/test feature sets
X_train = df_feat.loc[train_mask, extended_features]
X_test = df_feat.loc[test_mask, extended_features]

# %%
# Visualize portion of data for training and testing
plt.figure(figsize=(12,4))
plt.plot(df_feat.index, df_feat[target_onoff_col], label='Full Series', alpha=0.4)
plt.axvspan(train_start, train_end, color='green', alpha=0.3, label='Train Period')
plt.axvspan(test_start, test_end, color='orange', alpha=0.3, label='Test Period')
plt.title("Train-Test Split Visualization")
plt.xlabel("Time")
plt.ylabel("ON/OFF")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# %%
print("Train class distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTest class distribution:")
print(y_test.value_counts(normalize=True).round(3))

# %% [markdown]
# ## Train-Test Split for ON/OFF Model

# %%
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
# Calculate class imbalance weight
neg, pos = y_train.value_counts()
scale_pos_weight = neg / pos
print(f"Scale_pos_weight: {scale_pos_weight:.2f}")

# %%
# Initialize the classifier with class balancing
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight
)

# %%
# Train the model
model.fit(X_train, y_train)

# %%
# Predict ON/OFF
y_pred = model.predict(X_test)

# %%
# Evaluate model performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# %% [markdown]
# ## Regression model (predict consumption when ON)

# %%
# Filter data where appliance is ON (consumption > 0)
df_reg = df[df[f'{short_name}_onoff'] == 1].copy()

# %%
# Create time-based features for this subset
df_reg['hour'] = df_reg.index.hour
df_reg['dayofweek'] = df_reg.index.dayofweek
df_reg['quarter'] = df_reg.index.quarter
df_reg['month'] = df_reg.index.month
df_reg['year'] = df_reg.index.year
df_reg['dayofyear'] = df_reg.index.dayofyear
df_reg['dayofmonth'] = df_reg.index.day
df_reg['weekofyear'] = df_reg.index.isocalendar().week.astype(int)

# %%
# Define features and target for regression
feature_cols_reg = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
target_col_reg = short_name  # Actual consumption values

# %%
# Define train/test date ranges
train_mask_reg = (df_reg.index >= train_start) & (df_reg.index <= train_end)
test_mask_reg = (df_reg.index >= test_start) & (df_reg.index <= test_end)

X_train_reg = df_reg.loc[train_mask_reg, feature_cols_reg]
y_train_reg = df_reg.loc[train_mask_reg, target_col_reg]

X_test_reg = df_reg.loc[test_mask_reg, feature_cols_reg]
y_test_reg = df_reg.loc[test_mask_reg, target_col_reg]

# %%
# Check shapes
print(f"Regression train shape: {X_train_reg.shape}, test shape: {X_test_reg.shape}")
print(f"Regression train target mean: {y_train_reg.mean():.4f}, test target mean: {y_test_reg.mean():.4f}")

# %% [markdown]
# ## Train and Evaluate the Regression Model

# %%
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %%
# Initialize the XGBoost Regressor
reg_model = XGBRegressor(random_state=42, n_estimators=100, learning_rate=0.1)

# %%
# Train the model
reg_model.fit(X_train_reg, y_train_reg)

# %%
# Predict on test data
y_pred_reg = reg_model.predict(X_test_reg)

# %%
# Evaluate
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

print(f"Regression Model Evaluation:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# %% [markdown]
# ### Predicting

# %%
# Predict on test set
y_test_pred = reg_model.predict(X_test_reg)

# %%
# Create DataFrame to hold actual and predicted values
df_test_compare = pd.DataFrame({
    'Actual Consumption': y_test_reg,
    'Predicted Consumption': y_test_pred
}, index=X_test_reg.index)

# %%
# Plot actual vs predicted
plt.figure(figsize=(15,6))
plt.plot(df_test_compare.index, df_test_compare['Actual Consumption'], 
         label='Actual Consumption', color='blue')
plt.plot(df_test_compare.index, df_test_compare['Predicted Consumption'], 
         label='Predicted Consumption', color='orange', alpha=0.7)
plt.title(f'Actual vs Predicted {target_col} Consumption (Test Period - ON states)')
plt.xlabel('Timestamp')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
residuals = y_test_reg - y_test_pred
plt.figure(figsize=(10,5))
sns.histplot(residuals, bins=50, kde=True, color='orange')
plt.title('Distribution of Prediction Errors (Residuals)')
plt.xlabel('Error (Actual - Predicted)')
plt.grid(True)
plt.show()

# %%
print("Error summary:")
print(residuals.describe())

# %%
# Residuals statistics
residuals = y_test_reg - y_test_pred
print("Residuals summary:")
print(residuals.describe())

# %%
# Residuals correlation with actual values
corr = np.corrcoef(y_test_reg, residuals)[0,1]
print(f"\nCorrelation between actual and residuals: {corr:.4f}")

# %%
# Baseline model error comparison
baseline_pred = np.full_like(y_test_reg, y_train_reg.mean())
baseline_rmse = np.sqrt(np.mean((y_test_reg - baseline_pred)**2))
model_rmse = np.sqrt(np.mean((y_test_reg - y_test_pred)**2))
print(f"\nBaseline RMSE: {baseline_rmse:.4f}")
print(f"Model RMSE: {model_rmse:.4f}")
print(f"RMSE improvement over baseline: {baseline_rmse - model_rmse:.4f}")

# %% [markdown]
# ### Future prediction

# %%
# Future prediction window
future_start = pd.Timestamp("2016-07-08 00:00:00+00:00")
future_end = pd.Timestamp("2016-07-30 23:45:00+00:00")

df_future = df.copy()
df_future = df_future.loc[future_start:future_end].copy()

# %%
# Feature generation
df_future['hour'] = df_future.index.hour
df_future['dayofweek'] = df_future.index.dayofweek
df_future['quarter'] = df_future.index.quarter
df_future['month'] = df_future.index.month
df_future['year'] = df_future.index.year
df_future['dayofyear'] = df_future.index.dayofyear
df_future['dayofmonth'] = df_future.index.day
df_future['weekofyear'] = df_future.index.isocalendar().week.astype(int)

# %%
# Predict using trained model
X_future = df_future[feature_cols_reg]
df_future[f'Predicted_{short_name}'] = reg_model.predict(X_future)

# %%
print(df_future[[short_name, f'Predicted_{short_name}']].head())

# %%
plt.figure(figsize=(15,5))
plt.plot(df_future.index, df_future[short_name], label='Actual', alpha=0.7)
plt.plot(df_future.index, df_future[f'Predicted_{short_name}'], label='Predicted', alpha=0.7)
plt.title(f"Future Consumption: Actual vs Predicted ({target_col})")
plt.xlabel("Time")
plt.ylabel("Consumption (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
df_future['Error'] = df_future[short_name] - df_future[f'Predicted_{short_name}']
print(df_future['Error'].describe())

# %%
within_margin = (df_future['Error'].abs() <= 0.1).sum()
total_points = len(df_future)
print(f"Predictions within ±0.1 kWh: {within_margin} out of {total_points} ({(within_margin / total_points * 100):.2f}%)")

# %% [markdown]
# ### Weather Data Integration

# %%
import requests

# %%
# Define coordinates for the place (Essen)
latitude = 51.4576
longitude = 7.0225

# %%
# Define date range needed to extract weather data
start_date = "2016-01-01"
end_date = "2016-07-01"

# %%
# Define the weather parameters to fetch
parameters = [
    "temperature_2m", 
    "relative_humidity_2m", 
    "cloudcover", 
    "shortwave_radiation"
]

# %%
# Construct API URL
url = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    "&hourly=" + ",".join(parameters) +
    "&timezone=auto"
)

# %%
# Fetch data
response = requests.get(url)
data = response.json()

# %%
# Convert to DataFrame
weather_df = pd.DataFrame(data["hourly"])
weather_df['time'] = pd.to_datetime(weather_df['time'])
weather_df.set_index('time', inplace=True)

# %%
# Preview
print(weather_df.head())
print("\nColumns:", weather_df.columns.tolist())

# %% [markdown]
# ### Inspecting/Visualizing the Fetched Data

# %%
# Basic overview
print("Shape:", weather_df.shape)
print("Date range:", weather_df.index.min(), "to", weather_df.index.max())
print(weather_df.describe())

# %%
# Check missing values
print("Missing values per column:\n", weather_df.isnull().sum())

# %%
# Visualize weather trends
plt.figure(figsize=(15, 10))

# Temperature
plt.subplot(4, 1, 1)
weather_df['temperature_2m'].plot()
plt.title("Hourly Temperature (°C)")
plt.ylabel("°C")

# Humidity
plt.subplot(4, 1, 2)
weather_df['relative_humidity_2m'].plot()
plt.title("Hourly Relative Humidity (%)")
plt.ylabel("%")

# Cloud Cover
plt.subplot(4, 1, 3)
weather_df['cloudcover'].plot()
plt.title("Hourly Cloud Cover (%)")
plt.ylabel("%")

# Solar Radiation
plt.subplot(4, 1, 4)
weather_df['shortwave_radiation'].plot()
plt.title("Hourly Shortwave Radiation (W/m²)")
plt.ylabel("W/m²")

plt.xlabel("Date")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Filling the Weather data for 15 min format

# %%
# Resample to 15-minute intervals using forward-fill
weather_df_15min = weather_df.resample('15T').ffill()

# Check result
print("Resampled weather data shape:", weather_df_15min.shape)
print(weather_df_15min.head(8))

# %%
# Make both datetime indexes timezone-naive
df.index = df.index.tz_localize(None)
weather_df_15min.index = weather_df_15min.index.tz_localize(None)

# %%
# Align and merge weather with consumption data
start, end = '2016-01-01', '2016-07-01'
df = df.loc[start:end]
weather_df_15min = weather_df_15min.loc[start:end]

# Merge appliance consumption with weather on datetime index
df_merged = df.merge(weather_df_15min, left_index=True, right_index=True)

# Check result
print("Merged DataFrame shape:", df_merged.shape)
print(df_merged.head())

# %% [markdown]
# ### Visualize correlation patterns between consumption and weather

# %%
# Correlation matrix
correlation = df_merged.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix (Consumption vs Weather)")
plt.show()

# %%
# Consumption vs temperature
plt.figure(figsize=(15, 5))
plt.plot(df_merged.index, df_merged[short_name], 
         label=f'{target_col} Consumption (kWh)', alpha=0.7)
plt.plot(df_merged.index, df_merged['temperature_2m'], 
         label='Temperature (°C)', alpha=0.7)
plt.title("Consumption vs Temperature (Jan–June 2016)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Scatter plot
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df_merged, x='temperature_2m', y=short_name, alpha=0.5)
plt.title(f"{target_col} Consumption vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Consumption (kWh)")
plt.tight_layout()
plt.show()

# %%
# Resample to daily sum or mean
daily_consumption = df_merged[short_name].resample('D').sum()
daily_temperature = df_merged['temperature_2m'].resample('D').mean()

# Combine into one DataFrame
df_daily = pd.DataFrame({
    'daily_consumption': daily_consumption,
    'daily_temperature': daily_temperature
})

# Plot scatter
plt.figure(figsize=(6, 5))
sns.scatterplot(data=df_daily, x='daily_temperature', y='daily_consumption', alpha=0.7)
plt.title(f"Daily {target_col} Consumption vs Daily Average Temperature")
plt.xlabel("Daily Average Temperature (°C)")
plt.ylabel("Daily Total Consumption (kWh)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Feature Engineering (Weather related features)

# %%
df_feat = df_merged.copy()

# ON/OFF signal based on threshold
df_feat[f'{short_name}_onoff'] = (df_feat[short_name] > threshold).astype(int)

# Time features
df_feat['hour'] = df_feat.index.hour
df_feat['dayofweek'] = df_feat.index.dayofweek
df_feat['quarter'] = df_feat.index.quarter
df_feat['month'] = df_feat.index.month
df_feat['year'] = df_feat.index.year
df_feat['dayofyear'] = df_feat.index.dayofyear
df_feat['dayofmonth'] = df_feat.index.day
df_feat['weekofyear'] = df_feat.index.isocalendar().week.astype(int)

# Lag features for consumption
df_feat[f'{short_name}_lag1'] = df_feat[short_name].shift(1)
df_feat[f'{short_name}_lag2'] = df_feat[short_name].shift(2)
df_feat[f'{short_name}_lag4'] = df_feat[short_name].shift(4)
df_feat[f'{short_name}_lag96'] = df_feat[short_name].shift(96)

# Weather lag features
df_feat['temp_lag1'] = df_feat['temperature_2m'].shift(1)
df_feat['temp_lag4'] = df_feat['temperature_2m'].shift(4)
df_feat['temp_lag96'] = df_feat['temperature_2m'].shift(96)

# Rolling mean of consumption
df_feat[f'{short_name}_roll_1h'] = df_feat[short_name].rolling(window=4).mean()
df_feat[f'{short_name}_roll_1d'] = df_feat[short_name].rolling(window=96).mean()

# Drop rows with NaNs from shifting/rolling
df_feat.dropna(inplace=True)

# Inspect result
print("Feature set shape:", df_feat.shape)
print(df_feat[[short_name, f'{short_name}_lag1', 'temp_lag1', f'{short_name}_roll_1h']].head())

# %% [markdown]
# ### Train Test Split

# %%
# Remove timezone info from train/test timestamps
train_start = pd.Timestamp("2016-01-01 00:00:00").tz_localize(None)
train_end = pd.Timestamp("2016-06-30 23:45:00").tz_localize(None)
test_start = pd.Timestamp("2016-07-01 00:00:00").tz_localize(None)
test_end = pd.Timestamp("2016-07-07 23:45:00").tz_localize(None)

# Create boolean masks
train_mask = (df_feat.index >= train_start) & (df_feat.index <= train_end)
test_mask = (df_feat.index >= test_start) & (df_feat.index <= test_end)

# Define feature columns (excluding target)
feature_cols = [
    f'{short_name}_lag1', f'{short_name}_lag2', f'{short_name}_lag4', f'{short_name}_lag96',
    'temp_lag1', 'temp_lag4', 'temp_lag96',
    f'{short_name}_roll_1h', f'{short_name}_roll_1d',
    'temperature_2m', 'relative_humidity_2m', 'cloudcover', 'shortwave_radiation'
]

target_col = short_name

# %%
# Split features and target
X_train = df_feat.loc[train_mask, feature_cols]
y_train = df_feat.loc[train_mask, target_col]

X_test = df_feat.loc[test_mask, feature_cols]
y_test = df_feat.loc[test_mask, target_col]

# Check shapes and some info
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print(f"Training target mean: {y_train.mean():.4f}")
print(f"Testing target mean: {y_test.mean():.4f}")

# %% [markdown]
# ### Train Regression Model

# %%
from xgboost import XGBRegressor
import xgboost as xgb

# Define feature columns (all except target)
feature_cols = df_feat.columns.drop(short_name)

# Prepare data matrices
X_train = df_feat.loc[train_mask, feature_cols]
y_train = df_feat.loc[train_mask, short_name]
X_test = df_feat.loc[test_mask, feature_cols]
y_test = df_feat.loc[test_mask, short_name]

# %%
# Initialize XGBoost regressor
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# %%
# Train model
model.fit(X_train, y_train)

# %%
# Predict on test set
y_pred = model.predict(X_test)

# %%
# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

# %% [markdown]
# ### Visualize Results

# %%
plt.figure(figsize=(15, 5))

# Plot actual vs predicted
plt.plot(y_test.index, y_test.values, label='Actual Consumption', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted Consumption', color='orange', linestyle='--')

# Labels and title
plt.title(f"{target_col} Energy Consumption Prediction (15-min intervals)", fontsize=14)
plt.xlabel("Timestamp (15-minute intervals)", fontsize=12)
plt.ylabel("Energy Consumption (kWh per 15 min)", fontsize=12)

# Format x-axis
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Future Prediction

# %%
# Future prediction window
start_date = "2016-10-01"
end_date = "2016-10-14"

# Filter appliance data for window
df_appliance_window = df.loc[start_date:end_date].copy()
print(f"Appliance data shape between {start_date} and {end_date}:", df_appliance_window.shape)

if df_appliance_window.empty:
    raise ValueError(f"No appliance data available between {start_date} and {end_date}.")

# %%
# Feature generation for future prediction
df_appliance_window['hour'] = df_appliance_window.index.hour
df_appliance_window['dayofweek'] = df_appliance_window.index.dayofweek
df_appliance_window['quarter'] = df_appliance_window.index.quarter
df_appliance_window['month'] = df_appliance_window.index.month
df_appliance_window['year'] = df_appliance_window.index.year
df_appliance_window['dayofyear'] = df_appliance_window.index.dayofyear
df_appliance_window['dayofmonth'] = df_appliance_window.index.day
df_appliance_window['weekofyear'] = df_appliance_window.index.isocalendar().week.astype(int)

# %%
# Prepare features for prediction
X_future = df_appliance_window[feature_cols_reg]

# Predict using the trained regression model
df_appliance_window[f'Predicted_{short_name}'] = reg_model.predict(X_future)

# %%
# Plot results
plt.figure(figsize=(15,5))
plt.plot(df_appliance_window.index, df_appliance_window[short_name], 
         label='Actual Consumption', alpha=0.7)
plt.plot(df_appliance_window.index, df_appliance_window[f'Predicted_{short_name}'], 
         label='Predicted Consumption', alpha=0.7)
plt.title(f"{target_col} Consumption Prediction: {start_date} to {end_date}")
plt.xlabel("Time")
plt.ylabel("Consumption (kWh per 15 min)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Calculate errors
df_appliance