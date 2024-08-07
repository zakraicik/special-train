import pandas as pd
from datetime import datetime
import gzip
import logging
import pandas as pd
import numpy as np
from io import BytesIO, StringIO

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


def load_raw_data(aws_s3_client, bucket, key):

    response = aws_s3_client.get_object(Bucket=bucket, Key=key)

    logger.info("Downloading raw data from S3...")

    gzip_buffer = BytesIO(response["Body"].read())

    with gzip.GzipFile(fileobj=gzip_buffer, mode="rb") as gz_file:
        csv_content = gz_file.read().decode("utf-8")

    training_data = pd.read_csv(StringIO(csv_content))

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
