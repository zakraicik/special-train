import gzip
import logging
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from special_train.train.training_config import FEATURE_CONFIG, LAG_PERIODS, TARGET
from special_train.train.technical_indicators import technical_indicator_functions

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


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

    starting_n_columns = raw_data.shape[1]

    df = generate_technical_indicators(raw_data, FEATURE_CONFIG)

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

    model_features = features + [col for col in df.columns if "lag" in col]

    df[model_features] = df[model_features].diff()

    logger.info("Dropping rows with null")
    starting_n_rows = df.shape[0]

    df.dropna(inplace=True)

    ending_n_rows = df.shape[0]

    logger.info(f"Dropped {ending_n_rows - starting_n_rows} rows.")
    logger.info("Engineered dataset created.")

    return df, model_features


def validate_timestamps(df):
    df.index = pd.to_datetime(df.index, unit="ms")

    df.sort_index(inplace=True)

    time_diffs = df.index.to_series().diff().dropna()

    assert (
        time_diffs == pd.Timedelta(minutes=5)
    ).all(), "Not all rows are 5 minutes apart"

    return df


def split_data(df, train_size):

    assert 0 < train_size < 1, "train_size must be a float between 0 and 1"

    n = len(df)
    train_end = int(train_size * n)
    remainder = n - train_end
    val_end = train_end + remainder // 2

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def scale_datasets(train_df, test_df, val_df, feature_columns):

    scaler = MinMaxScaler()

    train_df.loc[:, feature_columns] = scaler.fit_transform(train_df[feature_columns])

    test_df.loc[:, feature_columns] = scaler.transform(test_df[feature_columns])
    val_df.loc[:, feature_columns] = scaler.transform(val_df[feature_columns])

    return train_df, test_df, val_df


def create_sequences(df, seq_length, target_column):
    X, y = [], []
    for i in range(len(df) - seq_length):
        X.append(df.iloc[i : i + seq_length].values)
        y.append(df.iloc[i + seq_length][target_column])
    return np.array(X), np.array(y)


def build_model(input_shape, output_shape):
    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(output_shape),
        ]
    )
    model.compile(optimizer=Adam(), loss="mse", metrics=["mae"])
    return model
