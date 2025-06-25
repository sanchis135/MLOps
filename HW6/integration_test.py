import os
import pandas as pd
from datetime import datetime
import batch_processing  

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def create_test_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    return pd.DataFrame(data, columns=columns)

def main():
    S3_ENDPOINT_URL = "http://localhost:4566"
    os.environ["S3_ENDPOINT_URL"] = S3_ENDPOINT_URL
    os.environ["INPUT_FILE_PATTERN"] = "s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"
    os.environ["OUTPUT_FILE_PATTERN"] = "s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"

    import boto3
    s3 = boto3.client("s3", endpoint_url=S3_ENDPOINT_URL)
    try:
        s3.create_bucket(Bucket="nyc-duration")
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass

    year, month = 2023, 1
    input_file = f"s3://nyc-duration/in/{year:04d}-{month:02d}.parquet"

    df_input = create_test_data()

    df_input.to_parquet(
        input_file,
        engine="pyarrow",
        compression=None,
        index=False,
        storage_options={"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}},
    )

    batch_processing.main()

    output_file = f"s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
    df_output = pd.read_parquet(
        output_file,
        storage_options={"client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}},
    )

    print("Result:")
    print(df_output)

    suma_predicciones = df_output["predicted_duration"].sum()
    print(f"Total of predictions: {suma_predicciones:.2f}")

if __name__ == "__main__":
    main()
