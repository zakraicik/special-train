import os
import boto3
from tensorflow.keras.callbacks import EarlyStopping

from special_train.train.training_config import SEQUENCE_LENGTH, TARGET
from special_train.train.train_utils import (
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

aws_access_key = os.environ.get("AWS_ACCESS_KEY")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

session = boto3.Session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_access_key,
    region_name=AWS_REGION,
)

aws_s3_client = session.client(service_name="s3")

if __name__ == "__main__":

    df = load_raw_data(aws_s3_client, S3_ETHEREUM_FORECAST_BUCKET, S3_ETHEREUM_PRICES)

    df, model_features = create_model_features(df)

    df = validate_timestamps(df)

    train_df, val_df, test_df = split_data(df, 0.8)

    train_df, val_df, test_df = scale_datasets(
        train_df, val_df, test_df, model_features
    )

    X_train, y_train = create_sequences(train_df, SEQUENCE_LENGTH, TARGET)

    X_val, y_val = create_sequences(val_df, SEQUENCE_LENGTH, TARGET)

    X_test, y_test = create_sequences(test_df, SEQUENCE_LENGTH, TARGET)

    model = build_model((SEQUENCE_LENGTH, len(model_features)), 1)

    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )

    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae}")
