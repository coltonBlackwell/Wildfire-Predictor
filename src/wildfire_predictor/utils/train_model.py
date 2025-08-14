import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib


def train(input_csv):
    """Train a single well-regularized model to predict wildfire size (log-transformed)."""

    # Load dataset
    df = pd.read_csv(input_csv)

    # Log-transform size to stabilize variance
    df['LOG_SIZE'] = np.log1p(df['SIZE_HA'])

    # Create extra features
    df['LAT_LONG'] = df['LATITUDE'] * df['LONGITUDE']
    df['SEASON'] = pd.cut(
        df['MONTH'],
        bins=[0, 3, 6, 9, 12],
        labels=['Winter', 'Spring', 'Summer', 'Fall']
    )
    df = pd.get_dummies(df, columns=['SEASON'], drop_first=True)

    # Feature selection
    feature_cols = [
        'LATITUDE', 'LONGITUDE', 'YEAR', 'MONTH', 'DAY', 'DAYOFYEAR', 'LAT_LONG',
        'CAUSE_H', 'CAUSE_L', 'CAUSE_U',
        'ECOZ_Boreal Cordillera', 'ECOZ_Boreal PLain', 'ECOZ_Montane Cordillera',
        'ECOZ_Pacific Maritime', 'ECOZ_Prairie', 'ECOZ_Taiga Plain',
        'SEASON_Spring', 'SEASON_Summer', 'SEASON_Fall'
    ]
    X = df[feature_cols].copy()
    y_reg = df['LOG_SIZE']

    # Scale numeric features
    numeric_cols = ['LATITUDE', 'LONGITUDE', 'YEAR', 'MONTH', 'DAY', 'DAYOFYEAR', 'LAT_LONG']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    # Single well-regularized regressor
    regressor = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=500,
        l2_regularization=0.1,
        random_state=42
    )
    regressor.fit(X_train, y_train)

    # Predictions in log space
    y_pred_log = regressor.predict(X_test)

    # Convert back to hectares
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)

    # Save model artifacts
    joblib.dump(
        (X_test, y_pred_log, y_test, regressor, scaler),
        'model_outputs.pkl'
    )

    return X, regressor, y_pred, y_test_actual
