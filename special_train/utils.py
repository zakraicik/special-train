import pandas as pd
import json
import logging
import pandas as pd

from datetime import datetime
from io import BytesIO
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def convert_millisecond_to_date(millisecond_timestamp):

    timestamp_in_seconds = millisecond_timestamp / 1000.0

    date_time = datetime.fromtimestamp(timestamp_in_seconds)

    return date_time


def convert_date_to_millisecond(date_time):

    timestamp_in_seconds = date_time.timestamp()

    millisecond_timestamp = int(timestamp_in_seconds * 1000)

    return millisecond_timestamp


def get_aws_secret(aws_secret_client, SecretId, secretName):
    try:
        get_secret_value_response = aws_secret_client.get_secret_value(
            SecretId=SecretId
        )
        secret = json.loads(get_secret_value_response["SecretString"])

        secret = secret[secretName]

        return secret

    except ClientError as e:
        raise e


def parquet_to_s3(df, bucket, key, aws_s3_client):
    out_buffer = BytesIO()
    df.to_parquet(out_buffer, index=True)

    out_buffer.seek(0)

    aws_s3_client.upload_fileobj(
        Fileobj=out_buffer,
        Bucket=bucket,
        Key=key,
        ExtraArgs={"ContentType": "binary/octet-stream"},
    )


def load_raw_data(aws_s3_client, bucket, key):

    logger.info("Downloading raw data from S3...")

    response = aws_s3_client.get_object(Bucket=bucket, Key=key)
    parquet_buffer = BytesIO(response["Body"].read())

    training_data = pd.read_parquet(parquet_buffer)

    logger.info(f"Dataset Size: {training_data.shape}")
    logger.info("Creating target...")

    training_data["next_period_close_change"] = (
        training_data["close"].pct_change().shift(-1)
    )

    logger.info("Reindexing... ")

    training_data.drop(columns=["otc"], inplace=True)
    training_data.dropna(inplace=True)

    training_data.set_index("timestamp", inplace=True)

    return training_data


def validate_timestamps(df):
    df.index = pd.to_datetime(df.index, unit="ms")

    df.sort_index(inplace=True)

    time_diffs = df.index.to_series().diff().dropna()

    assert (
        time_diffs == pd.Timedelta(minutes=5)
    ).all(), "Not all rows are 5 minutes apart"

    return df
