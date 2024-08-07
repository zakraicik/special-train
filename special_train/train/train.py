import os
import pandas as pd
import boto3
import gzip
import logging
from io import BytesIO, StringIO
from special_train.train.feature_engineering import generate_training_features
from special_train.train.config import FEATURE_CONFIG, RAW_COLUMNS, LAG_PERIODS, TARGET
from special_train.utils import validate_timestamps
from special_train.config import (
    AWS_REGION,
    S3_ETHEREUM_FORECAST_BUCKET,
    S3_ETHEREUM_PRICES,
)

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

aws_access_key = os.environ.get("AWS_ACCESS_KEY")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

session = boto3.Session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_access_key,
    region_name=AWS_REGION,
)

aws_s3_client = session.client(service_name="s3")

logger.info("S3 client created.")


def load_raw_data():

    response = aws_s3_client.get_object(
        Bucket=S3_ETHEREUM_FORECAST_BUCKET, Key=S3_ETHEREUM_PRICES
    )

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


def engineer_model_features(raw_data):

    logger.info("Creating technical indicators")

    starting_n_columns = raw_data.shape[1]

    df = generate_training_features(raw_data, FEATURE_CONFIG)

    ending_n_columns = df.shape[1]

    logger.info(f"Added {ending_n_columns - starting_n_columns} columns.")
    logger.info(f"Creating lagged columns ")

    starting_n_columns = df.shape[1]
    features = [x for x in df.columns if x != TARGET]
    lagged_features = []

    for column in features:
        for lag in LAG_PERIODS:
            lagged_feature = df[column].shift(lag)
            lagged_feature.name = f"{column}_lag_{lag}"
            lagged_features.append(lagged_feature)

    df = pd.concat([df] + lagged_features, axis=1)

    ending_n_columns = df.shape[1]

    logger.info(f"Added {ending_n_columns - starting_n_columns} columns.")
    logger.info("Differencing engineered features")

    df[features] = df[features].diff()

    logger.info("Dropping rows with null")
    starting_n_rows = df.shape[0]

    df.dropna(inplace=True)

    ending_n_rows = df.shape[0]

    logger.info(f"Dropped {ending_n_rows - starting_n_rows} rows.")
    logger.info("Engineered dataset created.")

    return df


def split_data(df, train_size):

    assert 0 < train_size < 1, "train_size must be a float between 0 and 1"

    n = len(df)
    train_end = int(train_size * n)
    remainder = n - train_end
    test_end = train_end + remainder // 2

    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end:test_end]
    val_df = df.iloc[test_end:]

    return train_df, test_df, val_df


if __name__ == "__main__":

    df = load_raw_data()

    df = engineer_model_features(df)

    df = validate_timestamps(df)

    train_df, test_df, val_df = split_data(df, 0.8)
