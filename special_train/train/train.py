import os
import boto3
from special_train.train.train_utils import (
    validate_timestamps,
    load_raw_data,
    create_model_features,
    split_data,
    scale_datasets,
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

    train_df, test_df, val_df = split_data(df, 0.8)

    train_df, test_df, val_df = scale_datasets(
        train_df, test_df, val_df, model_features
    )
