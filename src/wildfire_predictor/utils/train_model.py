import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

def train(input_csv):
    """Train a wildfire size model (log-transformed) with sample weighting for large fires,
       and preserve original LAT/LON for mapping."""

    # Load dataset
    df = pd.read_csv(input_csv)

    # Ensure SIZE_HA is positive
    df['SIZE_HA'] = df['SIZE_HA'].clip(lower=0.1)

    # Drop rows with missing key values
    # df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'SIZE_HA', 'MONTH', 'YEAR', 'DAY', 'DAYOFYEAR'])

    # Log-transform target to stabilize variance
    df['LOG_SIZE'] = np.log1p(df['SIZE_HA'])

    # Interaction feature
    df['LAT_LONG'] = df['LATITUDE'] * df['LONGITUDE']

    # Encode seasonality
    if 'MONTH' in df.columns:
        df['SEASON'] = pd.cut(
            df['MONTH'],
            bins=[0, 3, 6, 9, 12],
            labels=['Winter', 'Spring', 'Summer', 'Fall']
        )
        df = pd.get_dummies(df, columns=['SEASON'], drop_first=True)

    # Preserve original LAT/LON for mapping
    df['LAT_ORIG'] = df['LATITUDE']
    df['LON_ORIG'] = df['LONGITUDE']

    # Drop remaining string/categorical columns
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    df.drop(columns=non_numeric_cols, inplace=True, errors='ignore')

    # Feature-target split
    X = df.drop(columns=['SIZE_HA', 'LOG_SIZE'])
    X = X.fillna(0)  # Fill any remaining NaNs
    y_reg = df['LOG_SIZE']

    # Scale numeric features except LAT_ORIG / LON_ORIG
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    numeric_cols.remove('LAT_ORIG')
    numeric_cols.remove('LON_ORIG')
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Sample weights to emphasize large fires
    sample_weight = np.clip(df['SIZE_HA'] / df['SIZE_HA'].max(), 0.1, 1.0)

    # Train-test split
    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y_reg, sample_weight, test_size=0.2, random_state=42
    )

    # Train HistGradientBoostingRegressor
    regressor = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=800,
        l2_regularization=0.1,
        random_state=42
    )
    regressor.fit(X_train, y_train, sample_weight=sw_train)

    # Predict in log-space
    y_pred_log = regressor.predict(X_test)
    y_test_actual = np.expm1(y_test)
    y_pred = np.expm1(y_pred_log)

    # Save model artifacts
    joblib.dump((X_test, y_pred_log, y_test, regressor, scaler), 'model_outputs.pkl')

    print("Training complete. Model and outputs saved to 'model_outputs.pkl'.")
    return X, regressor, y_pred, y_test_actual
