import os
import logging
from boto3 import Session
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from special_train.utils import validate_timestamps, load_raw_data, parquet_to_s3
from special_train.config import (
    AWS_REGION,
    S3_ETHEREUM_FORECAST_BUCKET,
    S3_ETHEREUM_CONSOLIDATED_RAW_PRICE_DATA_KEY,
    S3_TRAIN_KEY,
    S3_VAL_KEY,
    S3_TEST_KEY,
    RAW_FEATURES,
)

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def split_data(df, train_size):

    logger.info("Splitting datasets...")

    assert 0 < train_size < 1, "train_size must be a float between 0 and 1"

    n = len(df)
    train_end = int(train_size * n)
    remainder = n - train_end
    val_end = train_end + remainder // 2

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def scale_datasets(train_df, test_df, val_df, model_features):

    scaler = MinMaxScaler()

    train_df.loc[:, model_features] = scaler.fit_transform(train_df[model_features])
    test_df.loc[:, model_features] = scaler.transform(test_df[model_features])
    val_df.loc[:, model_features] = scaler.transform(val_df[model_features])

    return train_df, test_df, val_df


if __name__ == "__main__":
    aws_access_key = os.environ.get("AWS_ACCESS_KEY")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    session = Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        region_name=AWS_REGION,
    )

    aws_s3_client = session.client(service_name="s3")

    df = load_raw_data(
        aws_s3_client,
        S3_ETHEREUM_FORECAST_BUCKET,
        S3_ETHEREUM_CONSOLIDATED_RAW_PRICE_DATA_KEY,
    )

    df = df.astype(float)

    df = validate_timestamps(df)

    logger.info(f"Creating Target")

    df["next_period_close_change"] = df["close"].pct_change().shift(-1)

    df.dropna(inplace=True)

    logger.info(f"Splitting Datasets")

    train_df, val_df, test_df = split_data(df, 0.8)

    logger.info(f"Scaling Datasets")

    train_df, val_df, test_df = scale_datasets(train_df, val_df, test_df, RAW_FEATURES)

    logger.info(
        f"Writing train_df to s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_TRAIN_KEY}"
    )

    parquet_to_s3(
        train_df,
        S3_ETHEREUM_FORECAST_BUCKET,
        S3_TRAIN_KEY,
        aws_s3_client,
    )

    logger.info(f"Writing val_df to s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_VAL_KEY}")
    parquet_to_s3(
        val_df,
        S3_ETHEREUM_FORECAST_BUCKET,
        S3_VAL_KEY,
        aws_s3_client,
    )

    logger.info(f"Writing test_df to s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_TEST_KEY}")
    parquet_to_s3(
        test_df,
        S3_ETHEREUM_FORECAST_BUCKET,
        S3_TEST_KEY,
        aws_s3_client,
    )
