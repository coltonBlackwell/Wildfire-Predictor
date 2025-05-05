import sys
import os
import pytest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from wildfire_predictor.utils.visualize.create_fire_grid import create_grid

@pytest.fixture
def sample_input_csv(tmp_path):

    data = {
        'LATITUDE': [49.0, 49.1, 49.2],
        'LONGITUDE': [-123.0, -123.1, -123.2],
        'YEAR': [2020, 2020, 2021]
    }
    df = pd.DataFrame(data)
    input_csv = tmp_path / "sample_input.csv"
    df.to_csv(input_csv, index=False)
    return input_csv

def test_create_grid(sample_input_csv, tmp_path):
    output_csv = tmp_path / "output_grid.csv"
    
    create_grid(str(sample_input_csv), str(output_csv), grid_size_km=10)

    assert output_csv.exists(), "Output CSV file was not created."

    df_out = pd.read_csv(output_csv)
    
    expected_columns = {'cell_id', 'YEAR', 'FIRE_COUNT', 'geometry'}

    assert expected_columns.issubset(set(df_out.columns)), f"Missing expected columns in output: {df_out.columns}"
    assert df_out['FIRE_COUNT'].sum() > 0, "No fire counts recorded."
    assert df_out['FIRE_COUNT'].sum() <= 3, "Unexpected number of fire counts."

