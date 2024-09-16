import os
import boto3
import numpy as np
import keras_tuner as kt
import tensorflow as tf

from datetime import datetime

from tensorflow.keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


from special_train.config import (
    AWS_REGION,
    S3_ETHEREUM_FORECAST_BUCKET,
    S3_X_TRAIN_KEY,
    S3_Y_TRAIN_KEY,
    S3_X_VAL_KEY,
    S3_Y_VAL_KEY,
    N,
)
from special_train.utils import load_numpy_from_s3, save_model_to_s3

aws_access_key = os.environ.get("AWS_ACCESS_KEY")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

session = boto3.Session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_access_key,
    region_name=AWS_REGION,
)

aws_secret_client = session.client(service_name="secretsmanager")
aws_s3_client = session.client(service_name="s3")

early_stopping = EarlyStopping(
    patience=10, monitor="val_loss", restore_best_weights=True
)

# reduce_lr = ReduceLROnPlateau(
#     monitor="val_loss",
#     factor=0.5,
#     patience=5,
#     min_lr=1e-6,
#     verbose=1,
# )


def build_model(hp):
    DROPOUT_RATE = hp.Float(
        "dropout_rate", min_value=0.1, max_value=0.5, sampling="linear"
    )
    LSTM_UNITS = hp.Choice("lstm_units", values=[16, 32, 64, 128])
    NUM_LAYERS = hp.Int("num_layers", min_value=1, max_value=5)
    l2_strength = hp.Float(
        "l2_strength", min_value=1e-6, max_value=1e-3, sampling="log"
    )

    model = Sequential()

    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))

    for i in range(NUM_LAYERS):
        model.add(
            LSTM(
                LSTM_UNITS,
                return_sequences=(i < NUM_LAYERS - 1),
            )
        )
        model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(LSTM_UNITS // 2, kernel_regularizer=l2(l2_strength)))
    model.add(Dense(N))

    learning_rate = hp.Float(
        "learning_rate", min_value=1e-5, max_value=1e-2, sampling="log"
    )

    optimizer = Adam(learning_rate=learning_rate, clipvalue=1)

    model.compile(
        optimizer=optimizer,
        loss="mae",
        metrics=["mae", "mse", "mape"],
    )

    return model


def tune_model(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    max_trials,
    executions_per_trial,
):

    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory="./special_train/train/tuning_logs/",
    )

    tuner.search(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

    return tuner, best_hp


if __name__ == "__main__":

    X_train = load_numpy_from_s3(
        aws_s3_client, S3_ETHEREUM_FORECAST_BUCKET, S3_X_TRAIN_KEY
    )

    y_train = load_numpy_from_s3(
        aws_s3_client, S3_ETHEREUM_FORECAST_BUCKET, S3_Y_TRAIN_KEY
    )

    X_val = load_numpy_from_s3(aws_s3_client, S3_ETHEREUM_FORECAST_BUCKET, S3_X_VAL_KEY)

    y_val = load_numpy_from_s3(aws_s3_client, S3_ETHEREUM_FORECAST_BUCKET, S3_Y_VAL_KEY)

    tuner, best_hp = tune_model(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        batch_size=32,
        max_trials=20,
        executions_per_trial=1,
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.compile(
        optimizer=Adam(learning_rate=best_hp.get("learning_rate"), clipvalue=1),
        loss="mae",
        metrics=["mae", "mse", "mape"],
    )
    save_model_to_s3(best_model, aws_s3_client, S3_ETHEREUM_FORECAST_BUCKET)
