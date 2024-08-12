import os
import logging
import pandas as pd
from boto3 import Session
from sklearn.preprocessing import MinMaxScaler

from special_train.config import TARGET
from special_train.utils import validate_timestamps, load_raw_data, parquet_to_s3
from special_train.config import (
    AWS_REGION,
    S3_ETHEREUM_FORECAST_BUCKET,
    S3_ETHEREUM_CONSOLIDATED_RAW_PRICE_DATA_KEY,
    S3_TRAIN_KEY,
    S3_VAL_KEY,
    S3_TEST_KEY,
    FEATURE_CONFIG,
    LAG_PERIODS,
)
from special_train.data.technical_indicators import technical_indicator_functions

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def generate_technical_indicators(df, config):
    new_features = {}
    for feature, settings in config.items():
        if feature in technical_indicator_functions:
            new_features = technical_indicator_functions[feature](
                df, settings, new_features
            )
    new_features_df = pd.DataFrame(new_features)
    df = pd.concat([df, new_features_df], axis=1)
    return df


def create_model_features(raw_data):
    logger.info("Creating technical indicators")

    df = generate_technical_indicators(raw_data, FEATURE_CONFIG)

    logger.info(f"Added {df.shape[1] - raw_data.shape[1]} columns.")
    logger.info("Creating lagged columns")

    features = [x for x in df.columns if x != TARGET]

    lagged_dfs = [
        df[features].shift(lag).add_suffix(f"_lag_{lag}") for lag in LAG_PERIODS
    ]

    df = pd.concat([df] + lagged_dfs, axis=1)

    logger.info(f"Added {df.shape[1] - len(features)} lagged columns.")
    logger.info("Differencing engineered features")

    model_features = features + [col for col in df.columns if "lag" in col]

    df[model_features] = df[model_features].diff()

    return df, model_features


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

    logger.info("Scaling datasets...")

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

    df, model_features = create_model_features(df)

    df = validate_timestamps(df)

    df.dropna(inplace=True)

    logger.info(f"Engineered Dataset Size: {df.shape}")

    train_df, val_df, test_df = split_data(df, 0.8)

    train_df, val_df, test_df = scale_datasets(
        train_df, val_df, test_df, model_features
    )

    # logger.info(
    #     f"Writing train_df to s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_TRAIN_KEY}"
    # )
    # parquet_to_s3(train_df, S3_ETHEREUM_FORECAST_BUCKET, S3_TRAIN_KEY, aws_s3_client)

    # logger.info(f"Writing val_df to s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_VAL_KEY}")
    # parquet_to_s3(val_df, S3_ETHEREUM_FORECAST_BUCKET, S3_VAL_KEY, aws_s3_client)

    # logger.info(f"Writing test_df to s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_TEST_KEY}")
    # parquet_to_s3(test_df, S3_ETHEREUM_FORECAST_BUCKET, S3_TEST_KEY, aws_s3_client)
