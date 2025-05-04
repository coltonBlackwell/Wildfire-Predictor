import sys
import os
import pytest
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from wildfire_predictor.utils.preprocessing import preprocess_wildfire_data

def write_csv(data, path):
    """Utility to write a DataFrame to CSV quickly"""
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def read_csv(path):
    """Read the output CSV and return a DataFrame"""
    return pd.read_csv(path)


def test_valid_data(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    data = {
        'SRC_AGENCY': ['BC'],
        'REP_DATE': ['2021-07-15'],
        'SIZE_HA': [100.123456],
        'LATITUDE': [53.0],
        'LONGITUDE': [-123.0],
        'CAUSE': ['Human'],
        'ECOZ_NAME': ['Boreal Plain'],
        'PROTZONE': ['Prince George']
    }
    write_csv(data, input_path)

    preprocess_wildfire_data(input_path, output_path)
    df_out = read_csv(output_path)

    assert len(df_out) == 1
    assert 'CAUSE_Human' in df_out.columns
    assert round(df_out['SIZE_HA'][0], 3) == 100.123


def test_drops_non_bc_rows(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    data = {
        'SRC_AGENCY': ['AB'],  # Not BC
        'REP_DATE': ['2021-07-15'],
        'SIZE_HA': [50],
        'LATITUDE': [53.0],
        'LONGITUDE': [-123.0],
        'CAUSE': ['Human'],
        'ECOZ_NAME': ['Taiga Plain'],
        'PROTZONE': ['Fort Nelson']
    }
    write_csv(data, input_path)
    preprocess_wildfire_data(input_path, output_path)

    df_out = read_csv(output_path)
    assert df_out.empty


def test_invalid_rep_date(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    data = {
        'SRC_AGENCY': ['BC'],
        'REP_DATE': ['not-a-date'],
        'SIZE_HA': [20],
        'LATITUDE': [50.0],
        'LONGITUDE': [-120.0],
        'CAUSE': ['Lightning'],
        'ECOZ_NAME': ['Pacific Maritime'],
        'PROTZONE': ['Kamloops']
    }
    write_csv(data, input_path)
    preprocess_wildfire_data(input_path, output_path)

    df_out = read_csv(output_path)
    assert df_out.empty


def test_non_summer_fire_dropped(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    data = {
        'SRC_AGENCY': ['BC'],
        'REP_DATE': ['2021-03-10'],
        'SIZE_HA': [20],
        'LATITUDE': [50.0],
        'LONGITUDE': [-120.0],
        'CAUSE': ['Human'],
        'ECOZ_NAME': ['Pacific Maritime'],
        'PROTZONE': ['Kamloops']
    }
    write_csv(data, input_path)
    preprocess_wildfire_data(input_path, output_path)

    df_out = read_csv(output_path)
    assert df_out.empty


def test_zero_size_ha_dropped(tmp_path):
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "output.csv"

    data = {
        'SRC_AGENCY': ['BC'],
        'REP_DATE': ['2021-06-10'],
        'SIZE_HA': [0],  # Invalid
        'LATITUDE': [50.0],
        'LONGITUDE': [-120.0],
        'CAUSE': ['Unknown'],
        'ECOZ_NAME': ['Montane Cordillera'],
        'PROTZONE': ['Cariboo']
    }
    write_csv(data, input_path)
    preprocess_wildfire_data(input_path, output_path)

    df_out = read_csv(output_path)
    assert df_out.empty
