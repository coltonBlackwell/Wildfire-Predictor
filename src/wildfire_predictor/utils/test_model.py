import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


def test(X, large_regressor, y_pred, y_test_actual):
    """ Test the model performance on the test set and visualize results."""

    mse = mean_squared_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)

    print(f"\n📊 Stratified Model Performance:")
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R^2: {r2:.2f}")

    # --- Plot 1: Predicted vs Actual ---
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_actual, y_pred, alpha=0.5)
    plt.plot([0, max(y_test_actual)], [0, max(y_test_actual)], 'r--', lw=2)
    plt.title('Predicted vs. Actual')
    plt.xlabel('Actual SIZE_HA')
    plt.ylabel('Predicted SIZE_HA')
    plt.show()

    # --- Plot 2: Residuals ---
    residuals = y_test_actual - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='blue', bins=50)
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.show()

    # --- Plot 3: Feature Importance ---
    plt.figure(figsize=(10, 6))
    xgb_importance = large_regressor.feature_importances_
    feature_names = X.columns

    sorted_idx = np.argsort(xgb_importance)[::-1]
    plt.barh(feature_names[sorted_idx], xgb_importance[sorted_idx], color='lightcoral')
    plt.title('Feature Importance from XGBoost (Large Fires Model)')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.show()