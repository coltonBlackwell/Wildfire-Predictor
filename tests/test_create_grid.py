import sys
import os
import pytest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from wildfire_predictor.utils.visualize.create_fire_grid import create_grid

@pytest.fixture
def sample_input_csv(tmp_path):
    # Create a small sample dataset
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
    
    # Run the function
    create_grid(str(sample_input_csv), str(output_csv), grid_size_km=10)

    # Check if output file was created
    assert output_csv.exists(), "Output CSV file was not created."

    # Load output and verify structure
    df_out = pd.read_csv(output_csv)
    
    # Check expected columns
    expected_columns = {'cell_id', 'YEAR', 'FIRE_COUNT', 'geometry'}
    assert expected_columns.issubset(set(df_out.columns)), f"Missing expected columns in output: {df_out.columns}"

    # Basic sanity checks
    assert df_out['FIRE_COUNT'].sum() > 0, "No fire counts recorded."
    assert df_out['FIRE_COUNT'].sum() <= 3, "Unexpected number of fire counts."

