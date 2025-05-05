import sys
import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from wildfire_predictor.utils.map import create_html_map


@pytest.fixture
def mock_dependencies(tmp_path):
    X_test = pd.DataFrame({
        'LATITUDE': [0.5] * 1000,
        'LONGITUDE': [0.5] * 1000,
        'CAUSE_H': [1] * 1000,
        'CAUSE_L': [0] * 1000,
        'CAUSE_U': [0] * 1000,
        'ECOZ_Boreal Cordillera': [0] * 1000,
        'ECOZ_Boreal PLain': [1] * 1000,
        'ECOZ_Montane Cordillera': [0] * 1000,
        'ECOZ_Pacific Maritime': [0] * 1000,
        'ECOZ_Prairie': [0] * 1000,
        'ECOZ_Taiga Plain': [0] * 1000,
    })
    y_pred_log = np.log1p([10] * 1000)
    small_idx_test = list(range(1000))
    y_reg_test = pd.Series(np.log1p([10] * 1000))
    y_class_pred = [0] * 1000

    mock_model = MagicMock()
    mock_model.predict.return_value = np.log1p([10])

    scaler = MagicMock()
    scaler.scale_ = [1, 1]
    scaler.mean_ = [50, -120]

    mock_pkl_data = (X_test, y_pred_log, small_idx_test, y_reg_test,
                     y_class_pred, mock_model, mock_model, scaler)

    geojson_path = tmp_path / "georef-canada-province@public.geojson"
    geojson_data = {
        "type": "FeatureCollection",
        "features": []
    }
    geojson_path.write_text(json.dumps(geojson_data), encoding="utf-8")

    return mock_pkl_data, str(geojson_path)


@patch("joblib.load")
@patch("builtins.open", new_callable=mock_open)
@patch("json.load")
def test_create_html_map(mock_json_load, mock_open_func, mock_joblib_load, mock_dependencies, tmp_path):
    mock_pkl_data, geojson_path = mock_dependencies
    mock_joblib_load.return_value = mock_pkl_data
    mock_json_load.return_value = {"type": "FeatureCollection", "features": []}

    with patch("folium.Map.save") as mock_save:
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_json_load.return_value))):
            os.chdir(tmp_path)
            create_html_map()

            mock_save.assert_called_once()