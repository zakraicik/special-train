import os
import logging
import talib
import numpy as np

from boto3 import Session
from sklearn.preprocessing import MinMaxScaler

from special_train.utils import (
    validate_timestamps,
    load_raw_data,
    save_numpy_to_s3,
    convert_millisecond_to_date,
)
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
    N,
    FEATURES,
)

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def split_data(df, train_size):

    assert 0 < train_size < 1, "train_size must be a float between 0 and 1"

    n = len(df)
    train_end = int(train_size * n)
    remainder = n - train_end
    val_end = train_end + int(remainder * (2 / 3))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def normalize_data(df, features, target):
    df.dropna(inplace=True)

    train_df, val_df, test_df = split_data(df, 0.70)

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_df[features] = feature_scaler.fit_transform(train_df[features])
    val_df[features] = feature_scaler.transform(val_df[features])
    test_df[features] = feature_scaler.transform(test_df[features])

    train_df[target] = target_scaler.fit_transform(train_df[[target]])
    val_df[target] = target_scaler.transform(val_df[[target]])
    test_df[target] = target_scaler.transform(test_df[[target]])

    return train_df, val_df, test_df, feature_scaler, target_scaler


def create_multistep_dataset(df, sequence_length, features, target, N):
    data = df[features].values
    target_data = df[target].values

    num_sequences = len(df) - sequence_length - N + 1

    sequences = np.array([data[i : i + sequence_length] for i in range(num_sequences)])

    targets = np.array(
        [
            target_data[i + sequence_length : i + sequence_length + N]
            for i in range(num_sequences)
        ]
    )

    return sequences, targets


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

    df.index = df.index.to_series().apply(convert_millisecond_to_date)

    logger.info(f"Filtering Records")

    df = df[df.index >= "2024-01-01"]

    logger.info(f"Creating Target")

    df.loc[:, "target"] = df.loc[:, "close"]

    logger.info(f"Splitting and Scaling Datasets")

    train_df, val_df, test_df, feature_scaler, target_scaler = normalize_data(
        df, FEATURES, "target"
    )

    X_train, y_train = create_multistep_dataset(train_df, 12, FEATURES, "target", N)
    X_val, y_val = create_multistep_dataset(val_df, 12, FEATURES, "target", N)
    X_test, y_test = create_multistep_dataset(test_df, 12, FEATURES, "target", N)

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
