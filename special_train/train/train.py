import os
import boto3
import argparse
from tensorflow.keras.callbacks import EarlyStopping

from special_train.train.training_config import SEQUENCE_LENGTH, TARGET
from special_train.train.utils import (
    validate_timestamps,
    load_raw_data,
    create_model_features,
    split_data,
    scale_datasets,
    create_sequences,
    build_model,
)

from special_train.config import (
    AWS_REGION,
    S3_ETHEREUM_FORECAST_BUCKET,
    S3_ETHEREUM_PRICES,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    aws_access_key = os.environ.get("AWS_ACCESS_KEY")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        region_name=AWS_REGION,
    )

    aws_s3_client = session.client(service_name="s3")

    model = build_model((SEQUENCE_LENGTH, len(model_features)), 1)

    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args["batch-size"],
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )

    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae}")
