import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib



def train(input_csv):
    """Train a model to predict wildfire size and classify large wildfires."""

    df = pd.read_csv(input_csv)

    df = df[df['SIZE_HA'] < df['SIZE_HA'].quantile(0.9)]
    df['IS_LARGE'] = (df['SIZE_HA'] > 1).astype(int)
    df['LOG_SIZE'] = np.log1p(df['SIZE_HA'])

    df['LAT_LONG'] = df['LATITUDE'] * df['LONGITUDE']
    df['SEASON'] = pd.cut(df['MONTH'], bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])
    df = pd.get_dummies(df, columns=['SEASON'], drop_first=True)

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

    numeric_cols = ['LATITUDE', 'LONGITUDE', 'YEAR', 'MONTH', 'DAY', 'DAYOFYEAR', 'LAT_LONG']
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )

    classifier = HistGradientBoostingClassifier(random_state=42)
    classifier.fit(X_train, y_class_train)

    small_idx_train = y_class_train == 0
    large_idx_train = y_class_train == 1

    small_regressor = HistGradientBoostingRegressor(random_state=42)
    large_regressor = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=10,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )

    small_regressor.fit(X_train[small_idx_train], y_reg_train[small_idx_train])
    large_regressor.fit(X_train[large_idx_train], y_reg_train[large_idx_train])

    y_class_pred = classifier.predict(X_test)

    y_pred_log = np.zeros_like(y_reg_test)
    small_idx_test = y_class_pred == 0
    large_idx_test = y_class_pred == 1
    y_pred_log[small_idx_test] = small_regressor.predict(X_test[small_idx_test])
    y_pred_log[large_idx_test] = large_regressor.predict(X_test[large_idx_test])

    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_reg_test)

    joblib.dump((X_test, y_pred_log, small_idx_test, y_reg_test, y_class_pred,
                small_regressor, large_regressor, scaler), 'model_outputs.pkl')

    return X, large_regressor, y_pred, y_test_actual