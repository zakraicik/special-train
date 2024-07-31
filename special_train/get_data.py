import os
import json
import boto3
import gzip
import pandas as pd
from polygon import RESTClient
from botocore.exceptions import ClientError
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from special_train.config import (
    AWS_REGION,
    S3_ETHEREUM_FORECAST_BUCKET,
)

aws_access_key = os.environ.get("AWS_ACCESS_KEY")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

session = boto3.Session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_access_key,
    region_name=AWS_REGION,
)

aws_secret_client = session.client(service_name="secretsmanager")
aws_s3_client = session.client(service_name="s3")


def get_aws_secret(SecretId):
    try:
        get_secret_value_response = aws_secret_client.get_secret_value(
            SecretId=SecretId
        )
        secret = json.loads(get_secret_value_response["SecretString"])

        polygon_api_key = secret["polygonApi"]

        return polygon_api_key

    except ClientError as e:
        raise e


def get_bucket_contents(bucket_name):

    contents = aws_s3_client.list_objects_v2(Bucket=bucket_name, Prefix="data")[
        "Contents"
    ]

    keys = [x["Key"] for x in contents]

    return keys


def create_polygon_client(api_key):
    polygon_client = RESTClient(api_key=api_key)
    return polygon_client


def get_prices(polygon_client, start_date, end_date):
    response = polygon_client.get_aggs(
        "X:ETHUSD", 5, "minute", start_date, end_date, limit=50000
    )

    df = pd.DataFrame(response)
    return df


def concat_training_data(bucket_name, keys):
    dataframes = []
    for key in keys:
        response = aws_s3_client.get_object(Bucket=bucket_name, Key=key)
        with gzip.GzipFile(fileobj=BytesIO(response["Body"].read())) as f:
            df = pd.read_csv(f)
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True).drop_duplicates(subset="timestamp")


if __name__ == "__main__":

    end_date = datetime.today() - timedelta(days=1)

    polygon_api_key = get_aws_secret("ethereum-price-forecast")

    polygon_client = create_polygon_client(polygon_api_key)

    df = get_prices(
        polygon_client,
        start_date=(end_date - timedelta(days=7)).strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    gzip_buffer = BytesIO()
    with gzip.GzipFile(mode="w", fileobj=gzip_buffer) as gz_file:
        gz_file.write(csv_buffer.getvalue().encode("utf-8"))

    aws_s3_client.put_object(
        Bucket=S3_ETHEREUM_FORECAST_BUCKET,
        Key=f"data/ethereum_prices_{(end_date - timedelta(days=7)).strftime('%Y_%m_%d')}_{end_date.strftime('%Y_%m_%d')}.csv.gz",
        Body=gzip_buffer.getvalue(),
    )

    keys = get_bucket_contents(S3_ETHEREUM_FORECAST_BUCKET)

    training_data = concat_training_data(S3_ETHEREUM_FORECAST_BUCKET, keys)

    assert (
        training_data["timestamp"].diff().iloc[1:] == 300000
    ).all(), "Inconsistent intervals in training data"

    csv_buffer = StringIO()
    training_data.to_csv(csv_buffer, index=False)

    gzip_buffer = BytesIO()
    with gzip.GzipFile(mode="w", fileobj=gzip_buffer) as gz_file:
        gz_file.write(csv_buffer.getvalue().encode("utf-8"))

    aws_s3_client.put_object(
        Bucket=S3_ETHEREUM_FORECAST_BUCKET,
        Key=f"data/ethereum_prices_all_time.csv.gz",
        Body=gzip_buffer.getvalue(),
    )
