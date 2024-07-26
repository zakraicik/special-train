import os
import json
import boto3
import pandas as pd
from polygon import RESTClient
from botocore.exceptions import ClientError
from io import StringIO
from datetime import datetime, timedelta
from special_train.config import (
    AWS_REGION,
    S3_ETHEREUM_FORECAST_BUCKET,
)

aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

session = boto3.session.Session(
    aws_access_key_id=aws_access_key_id,
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


def create_polygon_client(api_key):
    polygon_client = RESTClient(api_key=api_key)
    return polygon_client


def get_prices(polygon_client, start_date, end_date):
    response = polygon_client.get_aggs(
        "X:ETHUSD",
        1,
        "day",
        start_date,
        end_date,
        limit=50000,
    )

    df = pd.DataFrame(response)
    return df


if __name__ == "__main__":
    end_date = datetime.today() - timedelta(days=1)

    polygon_api_key = get_aws_secret("ethereum-price-forecast")

    polygon_client = create_polygon_client(polygon_api_key)

    df = get_prices(
        polygon_client,
        start_date=(end_date - timedelta(days=1)).strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    aws_s3_client.put_object(
        Bucket=S3_ETHEREUM_FORECAST_BUCKET,
        Key=f"data/ethereum_prices_{(end_date - timedelta(days=1)).strftime('%Y_%m_%d')}_{end_date.strftime('%Y_%m_%d')}",
        Body=csv_buffer.getvalue(),
    )
