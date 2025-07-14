import pandas as pd
from src.data.prepare_data import clean_data

def test_clean_data_returns_dataframe():
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
        "is_delay": [1, 0]   
    }

    df_raw = pd.DataFrame(sample_data)
    df_cleaned = clean_data(df_raw)

    assert isinstance(df_cleaned, pd.DataFrame)
    assert not df_cleaned.empty
    assert "is_delay" in df_cleaned.columns
    assert "CRSDepHour" in df_cleaned.columns
    assert df_cleaned["is_delay"].isin([0, 1]).all()
