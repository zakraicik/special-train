import os
import json
import boto3
import pandas as pd
from polygon import RESTClient
from io import BytesIO
from botocore.exceptions import ClientError
from datetime import datetime, timedelta
from special_train.utils import parquet_to_s3
from special_train.config import (
    AWS_REGION,
    SECRET_POLYGON_KEY,
    S3_ETHEREUM_FORECAST_BUCKET,
    S3_ETHEREUM_CONSOLIDATED_RAW_PRICE_DATA_KEY,
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


def get_bucket_contents(bucket_name, prefix, aws_s3_client):
    response = aws_s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    keys = [content["Key"] for content in response.get("Contents", [])]

    return [key for key in keys if not key.endswith("/")]


def create_polygon_client(api_key):
    polygon_client = RESTClient(api_key=api_key)
    return polygon_client


def get_prices(polygon_client, start_date, end_date):
    response = polygon_client.get_aggs(
        "X:ETHUSD", 5, "minute", start_date, end_date, limit=50000
    )

    df = pd.DataFrame(response)
    return df


def concat_training_data(bucket_name, keys, aws_s3_client):
    dataframes = []
    for key in keys:
        response = aws_s3_client.get_object(Bucket=bucket_name, Key=key)

        parquet_buffer = BytesIO(response["Body"].read())
        df = pd.read_parquet(parquet_buffer)

        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True).drop_duplicates(subset="timestamp")

    df.sort_values(by="timestamp", ascending=True, inplace=True)

    return df


if __name__ == "__main__":

    end_date = datetime.today() - timedelta(days=1)

    polygon_api_key = get_aws_secret(SECRET_POLYGON_KEY)

    polygon_client = create_polygon_client(polygon_api_key)

    df = get_prices(
        polygon_client,
        start_date=(end_date - timedelta(days=7)).strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    parquet_to_s3(
        df,
        S3_ETHEREUM_FORECAST_BUCKET,
        f"raw_data/ethereum_prices_{(end_date - timedelta(days=7)).strftime('%Y_%m_%d')}_{end_date.strftime('%Y_%m_%d')}.parquet",
        aws_s3_client,
    )

    keys = get_bucket_contents(S3_ETHEREUM_FORECAST_BUCKET, "raw_data", aws_s3_client)

    raw_data = concat_training_data(S3_ETHEREUM_FORECAST_BUCKET, keys, aws_s3_client)

    assert (
        raw_data["timestamp"].diff().iloc[1:] == 300000
    ).all(), "Inconsistent intervals in training data"

    parquet_to_s3(
        raw_data,
        S3_ETHEREUM_FORECAST_BUCKET,
        S3_ETHEREUM_CONSOLIDATED_RAW_PRICE_DATA_KEY,
        aws_s3_client,
    )
