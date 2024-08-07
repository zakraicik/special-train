import argparse
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from special_train.config import TARGET


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--valid", type=str, default=os.environ.get("SM_CHANNEL_VALID"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    return parser.parse_args()


def create_sequences(data, sequence_length, target_column):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data.iloc[i : (i + sequence_length)].values)
        y.append(data.iloc[i + sequence_length][target_column])
    return np.array(X), np.array(y)


def build_model(input_shape, lstm_units):
    model = Sequential(
        [
            LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(lstm_units // 2),
            Dropout(0.2),
            Dense(1),
        ]
    )
    return model


if __name__ == "__main__":
    args = parse_args()

    train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
    valid_df = pd.read_csv(os.path.join(args.valid, "valid.csv"))
    test_df = pd.read_csv(os.path.join(args.test, "test.csv"))

    feature_columns = [col for col in train_df.columns if col != TARGET]

    X_train, y_train = create_sequences(train_df, args.sequence_length, TARGET)
    X_valid, y_valid = create_sequences(valid_df, args.sequence_length, TARGET)
    X_test, y_test = create_sequences(test_df, args.sequence_length, TARGET)

    model = build_model((args.sequence_length, len(feature_columns)), args.lstm_units)
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate), loss="mse", metrics=["mae"]
    )

    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_valid, y_valid),
        callbacks=[early_stopping],
    )

    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae}")

    model.save(os.path.join(args.model_dir, "1"))
