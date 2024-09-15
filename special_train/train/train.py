import argparse
import os
import numpy as np

import tensorflow_io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping


DROPOUT_RATE = 0.2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument(
        "--train_x", type=str, default=os.environ.get("SM_CHANNEL_TRAIN_X")
    )
    parser.add_argument(
        "--train_y", type=str, default=os.environ.get("SM_CHANNEL_TRAIN_Y")
    )
    parser.add_argument(
        "--valid_x", type=str, default=os.environ.get("SM_CHANNEL_VALID_X")
    )
    parser.add_argument(
        "--valid_y", type=str, default=os.environ.get("SM_CHANNEL_VALID_Y")
    )

    return parser.parse_args()


def build_model(input_shape, lstm_units):
    model = Sequential()

    model.add(LSTM(lstm_units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(DROPOUT_RATE))

    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))

    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))

    model.add(LSTM(lstm_units, return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))

    model.add(LSTM(lstm_units))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(lstm_units // 2))

    model.add(Dense(1))

    return model


if __name__ == "__main__":
    args = parse_args()

    X_train = np.load(os.path.join(args.train_x, "X_train.npy"))
    y_train = np.load(os.path.join(args.train_y, "y_train.npy"))

    X_valid = np.load(os.path.join(args.valid_x, "X_val.npy"))
    y_valid = np.load(os.path.join(args.valid_y, "y_val.npy"))

    input_shape = (X_train.shape[1], X_train.shape[2])

    model = build_model(input_shape, args.lstm_units)

    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss="mae",
        metrics=["mae", "mse", "mape"],
    )

    early_stopping = EarlyStopping(
        patience=10, monitor="val_loss", restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stopping],
    )

    model.save(args.model_dir)
