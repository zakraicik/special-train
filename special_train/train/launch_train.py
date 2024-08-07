import os
import sagemaker
import boto3
from sagemaker.tensorflow import TensorFlow
from special_train.config import (
    S3_ETHEREUM_FORECAST_BUCKET,
    S3_TRAIN_KEY,
    S3_TEST_KEY,
    S3_VAL_KEY,
    AWS_REGION,
)


def launch_training():
    aws_access_key = os.environ.get("AWS_ACCESS_KEY")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        region_name=AWS_REGION,
    )

    sagemaker_session = sagemaker.Session(boto_session=session)

    role = "sagemaker_role_arn"

    estimator = TensorFlow(
        entry_point="train.py",
        source_dir="./special_train",
        role=role,
        instance_count=1,
        instance_type="ml.c5.xlarge",
        framework_version="2.6.0",
        py_version="py38",
        hyperparameters={
            "epochs": 100,
            "batch-size": 32,
            "learning-rate": 0.001,
            "lstm-units": 64,
            "sequence-length": 24,
        },
    )

    estimator.fit(
        {
            "train": f"s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_TRAIN_KEY}",
            "valid": f"s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_VAL_KEY}",
            "test": f"s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_TEST_KEY}",
        }
    )


if __name__ == "__main__":
    launch_training()
