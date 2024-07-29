import os
import pandas as pd
import boto3
import gzip
from io import BytesIO
from datetime import datetime
from special_train.config import AWS_REGION, S3_ETHEREUM_FORECAST_BUCKET

aws_access_key = os.environ.get("AWS_ACCESS_KEY")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

session = boto3.Session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_access_key,
    region_name=AWS_REGION,
)

aws_s3_client = session.client(service_name="s3")


def convert_millisecond_to_date(millisecond_timestamp):

    timestamp_in_seconds = millisecond_timestamp / 1000.0

    date_time = datetime.fromtimestamp(timestamp_in_seconds)

    return date_time


def get_bucket_contents(bucket_name):

    contents = aws_s3_client.list_objects_v2(Bucket=bucket_name, Prefix="data")[
        "Contents"
    ]

    keys = [x["Key"] for x in contents]

    return keys


def download_and_extract_csv(bucket_name, keys):
    dataframes = []
    for key in keys:
        response = aws_s3_client.get_object(Bucket=bucket_name, Key=key)
        with gzip.GzipFile(fileobj=BytesIO(response["Body"].read())) as f:
            df = pd.read_csv(f)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True).drop_duplicates(subset="timestamp")


if __name__ == "__main__":

    keys = get_bucket_contents(S3_ETHEREUM_FORECAST_BUCKET)

    df = download_and_extract_csv(S3_ETHEREUM_FORECAST_BUCKET, keys)

    assert (
        df["timestamp_diff"].iloc[1:] == 300000
    ).all(), "Inconsistent intervals in training data"
