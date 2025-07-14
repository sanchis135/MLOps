import pandas as pd
from sklearn.model_selection import train_test_split
import os

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate encoded features for the model.
    """
    df = df.copy()

    # Extract scheduled hour from CRSDepTime if exists (format HHMM)
    if 'CRSDepTime' in df.columns:
        df['ScheduledHour'] = df['CRSDepTime'].astype(str).str.zfill(4).str[:2].astype(int)

    # One-hot encode categorical variables
    categorical_cols = ['Reporting_Airline', 'Origin', 'Dest', 'OriginState', 'DestState']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Drop unnecessary columns (including original categorical columns and target if present)
    cols_to_drop = ['ArrDelay', 'ArrDelayMinutes', 'FlightDate', 'CRSDepTime']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    return df

def split_and_save_data(df: pd.DataFrame, target_col='is_delay',
                        test_size=0.2, random_state=42):
    """
    Split into train/test and save to disk.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("✅ Train and test sets saved in data/processed/")

if __name__ == "__main__":
    # Load clean data
    input_path = "data/processed/flights_clean.csv"
    df_clean = pd.read_csv(input_path)

    # Ensure target is int
    if df_clean['is_delay'].dtype != int:
        df_clean['is_delay'] = df_clean['is_delay'].astype(int)

    # Build features
    df_features = build_features(df_clean)

    # Split and save
    split_and_save_data(df_features)
