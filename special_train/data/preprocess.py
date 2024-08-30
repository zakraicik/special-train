import os
import logging
from boto3 import Session
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from special_train.utils import validate_timestamps, load_raw_data, save_numpy_to_s3
from special_train.config import (
    AWS_REGION,
    S3_ETHEREUM_FORECAST_BUCKET,
    S3_ETHEREUM_CONSOLIDATED_RAW_PRICE_DATA_KEY,
    S3_X_TRAIN_KEY,
    S3_Y_TRAIN_KEY,
    S3_X_VAL_KEY,
    S3_Y_VAL_KEY,
    S3_X_TEST_KEY,
    S3_Y_TEST_KEY,
    WINDOW_LENGTH,
    FEATURES,
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


def normalize_data(df, features, target_column):

    price_column = "close"

    df[price_column] = df[price_column] / df[price_column].iloc[0] - 1

    df = df[features + [target_column]]

    train_df, val_df, test_df = split_data(df, 0.8)

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    scaler = MinMaxScaler()

    train_df[features] = scaler.fit_transform(train_df[features])

    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])

    return train_df, val_df, test_df


def create_sliding_windows(df, feature_columns, target_column, win_len=10):

    features = df[feature_columns].values
    targets = df[target_column].values

    X, y = [], []

    for i in range(len(df) - win_len):

        X_window = features[i : i + win_len]
        y_value = targets[i + win_len]

        X.append(X_window)
        y.append(y_value)

    X = np.array(X)
    y = np.array(y)

    y = y.reshape(-1)

    return X, y


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

    df["target"] = df["close"]

    df.dropna(inplace=True)

    logger.info(f"Splitting and Scaling Datasets")

    train_df, val_df, test_df = normalize_data(df, FEATURES, "target")

    X_train, y_train = create_sliding_windows(
        train_df, FEATURES, "target", WINDOW_LENGTH
    )

    X_val, y_val = create_sliding_windows(val_df, FEATURES, "target", WINDOW_LENGTH)

    X_test, y_test = create_sliding_windows(test_df, FEATURES, "target", WINDOW_LENGTH)

    logger.info(
        f"Writing X_train to s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_X_TRAIN_KEY}"
    )

    save_numpy_to_s3(
        aws_s3_client, X_train, S3_ETHEREUM_FORECAST_BUCKET, S3_X_TRAIN_KEY
    )

    logger.info(
        f"Writing y_train to s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_Y_TRAIN_KEY}"
    )

    save_numpy_to_s3(
        aws_s3_client, y_train, S3_ETHEREUM_FORECAST_BUCKET, S3_Y_TRAIN_KEY
    )

    logger.info(f"Writing X_val to s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_X_VAL_KEY}")

    save_numpy_to_s3(aws_s3_client, X_val, S3_ETHEREUM_FORECAST_BUCKET, S3_X_VAL_KEY)

    logger.info(f"Writing y_val to s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_Y_VAL_KEY}")

    save_numpy_to_s3(aws_s3_client, y_val, S3_ETHEREUM_FORECAST_BUCKET, S3_Y_VAL_KEY)

    logger.info(f"Writing X_test to s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_X_TEST_KEY}")

    save_numpy_to_s3(aws_s3_client, X_test, S3_ETHEREUM_FORECAST_BUCKET, S3_X_TEST_KEY)

    logger.info(f"Writing y_test to s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_Y_TEST_KEY}")

    save_numpy_to_s3(aws_s3_client, y_test, S3_ETHEREUM_FORECAST_BUCKET, S3_Y_TEST_KEY)
