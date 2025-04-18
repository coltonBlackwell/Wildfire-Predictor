import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

# Load the dataset
df = pd.read_csv('../data/wildfires_preprocessed.csv')  # Replace with actual path

# Filter the data
df = df[df['SIZE_HA'] < df['SIZE_HA'].quantile(0.99)]

# Target variable (log-transform to handle skew)
y = np.log1p(df['SIZE_HA'])

# Feature engineering
df['LAT_LONG'] = df['LATITUDE'] * df['LONGITUDE']
df['SEASON'] = pd.cut(df['MONTH'], bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])
df = pd.get_dummies(df, columns=['SEASON'], drop_first=True)

# Features for model
feature_cols = [
    'LATITUDE', 'LONGITUDE', 'YEAR', 'MONTH', 'DAY', 'DAYOFYEAR', 'LAT_LONG',
    'CAUSE_H', 'CAUSE_L', 'CAUSE_U',
    'ECOZ_Boreal Cordillera', 'ECOZ_Boreal PLain', 'ECOZ_Montane Cordillera',
    'ECOZ_Pacific Maritime', 'ECOZ_Prairie', 'ECOZ_Taiga Plain',
    'SEASON_Spring', 'SEASON_Summer', 'SEASON_Fall'
]
X = df[feature_cols].copy()

# Normalize numeric columns
numeric_cols = ['LATITUDE', 'LONGITUDE', 'YEAR', 'MONTH', 'DAY', 'DAYOFYEAR', 'LAT_LONG']
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
hist_model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=31,
    max_depth=10,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    random_state=42
)

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

# Ensemble model
ensemble = VotingRegressor(estimators=[
    ('histgb', hist_model),
    ('xgb', xgb_model)
])

# Train the model
ensemble.fit(X_train, y_train)

# Make predictions
y_pred_log = ensemble.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

# Evaluation metrics
mse = mean_squared_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

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
xgb_model.fit(X_train, y_train)  # Fit XGBoost model separately for feature importance
xgb_importance = xgb_model.feature_importances_
feature_names = X.columns

# Sort the feature importances
sorted_idx = np.argsort(xgb_importance)[::-1]
sorted_importances = xgb_importance[sorted_idx]
sorted_feature_names = feature_names[sorted_idx]

plt.barh(sorted_feature_names, sorted_importances, color='lightcoral')
plt.title('Feature Importance from XGBoost')
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.show()
