import sys
import os
import pytest
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from wildfire_predictor.utils.train_model import train

@pytest.fixture
def synthetic_dataset(tmp_path):
    # Create a fake but valid dataset with key dummy feature columns used by `train()`
    data = {
        'SIZE_HA': [0.5, 2.0, 3.5, 0.2, 0.8, 1.2, 4.5, 2.3, 0.1, 1.8],
        'LATITUDE': [50.1]*10,
        'LONGITUDE': [-120.3]*10,
        'YEAR': [2021]*10,
        'MONTH': [6]*10,
        'DAY': [15]*10,
        'DAYOFYEAR': [166]*10,
        'CAUSE_H': [1,0,0,1,0,1,0,1,0,0],
        'CAUSE_L': [0,1,0,0,1,0,1,0,1,0],
        'CAUSE_U': [0,0,1,0,0,0,0,0,0,1],
        'ECOZ_Boreal Cordillera': [0]*10,
        'ECOZ_Boreal PLain': [1]*10,
        'ECOZ_Montane Cordillera': [0]*10,
        'ECOZ_Pacific Maritime': [0]*10,
        'ECOZ_Prairie': [0]*10,
        'ECOZ_Taiga Plain': [0]*10
    }

    df = pd.DataFrame(data)
    input_csv = tmp_path / "train_input.csv"
    df.to_csv(input_csv, index=False)
    return input_csv


def test_train_returns_valid_outputs(synthetic_dataset):
    X, model, y_pred, y_test_actual = train(synthetic_dataset)

    # Check shapes
    assert X.shape[0] >= 1
    assert len(y_pred) == len(y_test_actual)
    
    # Check model type
    from xgboost import XGBRegressor
    assert isinstance(model, XGBRegressor)

    # Check prediction values are finite
    assert np.isfinite(y_pred).all()
    assert np.isfinite(y_test_actual).all()


def test_model_artifacts_saved(synthetic_dataset):
    _ = train(synthetic_dataset)
    assert os.path.exists("model_outputs.pkl")

    with open("model_outputs.pkl", "rb") as f:
        contents = joblib.load(f)

    # Unpack and check structure
    (X_test, y_pred_log, small_idx_test, y_reg_test, y_class_pred,
     small_regressor, large_regressor, scaler) = contents

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    assert isinstance(small_regressor, HistGradientBoostingRegressor)
    assert isinstance(large_regressor, XGBRegressor)
    assert isinstance(scaler, StandardScaler)
    assert isinstance(X_test, pd.DataFrame)
    assert len(y_pred_log) == len(y_class_pred)

