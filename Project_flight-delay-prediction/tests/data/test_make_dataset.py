import pandas as pd
import os
from src.data.make_dataset import build_features, split_and_save_data

def test_build_features():
    sample_data = {
        "FlightDate": ["2018-01-01", "2018-01-02"],
        "Cancelled": [0, 1],
        "Diverted": [0, 0],
        "DepDelay": [5.0, None],
        "ArrDelay": [10.0, None],
        "ArrDelayMinutes": [10.0, None],
        "Carrier": ["AA", "DL"],
        "Origin": ["JFK", "LAX"],
        "Dest": ["LAX", "JFK"],
        "CRSDepTime": [1400, 1500],
        "Distance": [2475.0, 2475.0],
        "AirTime": [300.0, None],
        "Reporting_Airline": ["AA", "DL"],       
        "OriginState": ["NY", "CA"],             
        "DestState": ["CA", "NY"]                
    }

    df = pd.DataFrame(sample_data)
    df_features = build_features(df)

    assert 'ArrDelay' not in df_features.columns
    assert 'CRSDepTime' not in df_features.columns
    assert 'FlightDate' not in df_features.columns

    assert any(col.startswith('Reporting_Airline_') for col in df_features.columns)
    assert 'ScheduledHour' in df_features.columns

def test_split_and_save_data(tmp_path):
    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        df = pd.DataFrame({
            "feature1": list(range(10)),
            "feature2": list(range(10, 110, 10)),
            "is_delay": [0, 1] * 5
        })

        split_and_save_data(df, test_size=0.3)

        train_file = tmp_path / "data" / "processed" / "train.csv"
        test_file = tmp_path / "data" / "processed" / "test.csv"

        assert train_file.exists()
        assert test_file.exists()

        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)

        assert 'is_delay' in df_train.columns
        assert 'is_delay' in df_test.columns
    finally:
        os.chdir(original_cwd)
