import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the dataset
df = pd.read_csv('../data/wildfires_preprocessed.csv')  # Adjust path as needed
df = df[df['SIZE_HA'] < df['SIZE_HA'].quantile(0.9)]  # Remove top 1% outliers

# Binary label for classification (small ≤ 1 ha, large > 1 ha)
df['IS_LARGE'] = (df['SIZE_HA'] > 1).astype(int)

# Log-transform target for regression
df['LOG_SIZE'] = np.log1p(df['SIZE_HA'])

# Feature engineering
df['LAT_LONG'] = df['LATITUDE'] * df['LONGITUDE']
df['SEASON'] = pd.cut(df['MONTH'], bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])
df = pd.get_dummies(df, columns=['SEASON'], drop_first=True)

# Feature columns
feature_cols = [
    'LATITUDE', 'LONGITUDE', 'YEAR', 'MONTH', 'DAY', 'DAYOFYEAR', 'LAT_LONG',
    'CAUSE_H', 'CAUSE_L', 'CAUSE_U',
    'ECOZ_Boreal Cordillera', 'ECOZ_Boreal PLain', 'ECOZ_Montane Cordillera',
    'ECOZ_Pacific Maritime', 'ECOZ_Prairie', 'ECOZ_Taiga Plain',
    'SEASON_Spring', 'SEASON_Summer', 'SEASON_Fall'
]
X = df[feature_cols].copy()
y_class = df['IS_LARGE']
y_reg = df['LOG_SIZE']

# Normalize numeric features
numeric_cols = ['LATITUDE', 'LONGITUDE', 'YEAR', 'MONTH', 'DAY', 'DAYOFYEAR', 'LAT_LONG']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Split the data
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_class, y_reg, test_size=0.2, random_state=42
)

# Train classifier
classifier = HistGradientBoostingClassifier(random_state=42)
classifier.fit(X_train, y_class_train)

# Split training data into small and large for regression
small_idx_train = y_class_train == 0
large_idx_train = y_class_train == 1

# Define regressors
small_regressor = HistGradientBoostingRegressor(random_state=42)
large_regressor = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

# Train regressors on respective subsets
small_regressor.fit(X_train[small_idx_train], y_reg_train[small_idx_train])
large_regressor.fit(X_train[large_idx_train], y_reg_train[large_idx_train])

# Classify test set
y_class_pred = classifier.predict(X_test)

# Allocate predictions based on classification
y_pred_log = np.zeros_like(y_reg_test)

# Small fire predictions
small_idx_test = y_class_pred == 0
y_pred_log[small_idx_test] = small_regressor.predict(X_test[small_idx_test])

# Large fire predictions
large_idx_test = y_class_pred == 1
y_pred_log[large_idx_test] = large_regressor.predict(X_test[large_idx_test])

# Convert back to real SIZE_HA
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_reg_test)

# Evaluate
mse = mean_squared_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print(f"\n📊 Stratified Model Performance:")
print(f"Test MSE: {mse:.2f}")
print(f"Test R^2: {r2:.2f}")

# Visualization 1: Predicted vs Actual Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test_actual, y_pred, alpha=0.5)
plt.plot([0, max(y_test_actual)], [0, max(y_test_actual)], 'r--', lw=2)  # Diagonal line
plt.title('Predicted vs. Actual')
plt.xlabel('Actual SIZE_HA')
plt.ylabel('Predicted SIZE_HA')
plt.show()

# Visualization 2: Residuals Plot
residuals = y_test_actual - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='blue', bins=50)
plt.title('Residuals Distribution')
plt.xlabel('Residuals (Actual - Predicted)')
plt.ylabel('Frequency')
plt.show()

# Visualization 3: Feature Importance Plot (from XGBoost model)
plt.figure(figsize=(10, 6))
xgb_importance = large_regressor.feature_importances_  # Use XGBoost model
feature_names = X.columns

# Sort the feature importances
sorted_idx = np.argsort(xgb_importance)[::-1]
sorted_importances = xgb_importance[sorted_idx]
sorted_feature_names = feature_names[sorted_idx]

plt.barh(sorted_feature_names, sorted_importances, color='lightcoral')
plt.title('Feature Importance from XGBoost (Large Fires Model)')  # Updated title
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()

# ======= NEW SECTION: Inspect one prediction =======

# Pick a test sample (first row)
sample = X_test.iloc[0:1]
sample_pred_log = small_regressor.predict(sample)
sample_pred = np.expm1(sample_pred_log)

# Actual value
actual_log = y_pred_log[small_idx_test][0]
actual = np.expm1(actual_log)

print("\n--- Single Prediction Example ---")
print(f"Predicted log(SIZE_HA): {sample_pred_log[0]:.4f}")
print(f"Predicted SIZE_HA: {sample_pred[0]:.2f}")
print(f"Actual log(SIZE_HA): {actual_log:.4f}")
print(f"Actual SIZE_HA: {actual:.2f}")