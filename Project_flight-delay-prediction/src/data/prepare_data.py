import pandas as pd
import os

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and prepares the original flights dataset.
    """
    df = df.copy()

    # Convert 'FlightDate' to datetime
    df['FlightDate'] = pd.to_datetime(df['FlightDate'])

    # Extract hour from CRSDepTime
    df['CRSDepHour'] = df['CRSDepTime'].astype(str).str.zfill(4).str[:2].astype(int)

    # Fill null values in key columns
    df['ArrDelayMinutes'] = df['ArrDelayMinutes'].fillna(0)
    df['ArrDelay'] = df['ArrDelay'].fillna(0)
    df['AirTime'] = df['AirTime'].fillna(df['AirTime'].median())

    # Ensure is_delay is a binary int
    df['is_delay'] = df['is_delay'].astype(int)

    # Drop unnecessary or duplicate columns if desired (example)
    # df.drop(columns=['CRSDepTime'], inplace=True)

    return df

def main():
    input_path = "data/raw/archive/flight_delay_predict.csv"  # Change according to your original dataset
    output_path = "data/processed/flights_clean.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load raw data
    df = pd.read_csv(input_path)

    # Clean data
    df_clean = clean_data(df)

    # Save cleaned dataset
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to {output_path}")

if __name__ == "__main__":
    main()
