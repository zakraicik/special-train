import os
import logging
import talib
import numpy as np

from boto3 import Session
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


def add_technical_features(df):

    df.loc[:, "SMA_10"] = talib.SMA(df["close"], timeperiod=10)
    df.loc[:, "SMA_30"] = talib.SMA(df["close"], timeperiod=30)

    return df


def normalize_data(df, features, target_column):

    sma_columns = ["SMA_10", "SMA_30"]

    price_column = "close"

    df[price_column] = df[price_column] / df[price_column].iloc[0] - 1
    df["SMA_10"] = df["SMA_10"] / df["SMA_10"].iloc[0] - 1
    df["SMA_30"] = df["SMA_30"] / df["SMA_30"].iloc[0] - 1

    df = df[features + sma_columns + [target_column]]

    train_df, val_df, test_df = split_data(df, 0.8)

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    scaler = MinMaxScaler()

    train_df[features + sma_columns] = scaler.fit_transform(
        train_df[features + sma_columns]
    )

    val_df[features + sma_columns] = scaler.transform(val_df[features + sma_columns])
    test_df[features + sma_columns] = scaler.transform(test_df[features + sma_columns])

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

    logger.info(f"Creating Technical Features")

    df = add_technical_features(df)

    df.dropna(inplace=True)

    logger.info(f"Splitting and Scaling Datasets")

    train_df, val_df, test_df = normalize_data(df, FEATURES, "target")

    X_train, y_train = create_sliding_windows(
        train_df, FEATURES + ["SMA_10", "SMA_30"], "target", WINDOW_LENGTH
    )

    X_val, y_val = create_sliding_windows(
        val_df, FEATURES + ["SMA_10", "SMA_30"], "target", WINDOW_LENGTH
    )

    X_test, y_test = create_sliding_windows(
        test_df, FEATURES + ["SMA_10", "SMA_30"], "target", WINDOW_LENGTH
    )

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
