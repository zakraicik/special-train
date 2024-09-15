import os
import sagemaker
import boto3
from sagemaker.tensorflow import TensorFlow
from special_train.config import (
    S3_ETHEREUM_FORECAST_BUCKET,
    SECRET_ID,
    SECRET_SAGEMAKER_ARN_KEY,
    S3_X_TRAIN_KEY,
    S3_Y_TRAIN_KEY,
    S3_X_VAL_KEY,
    S3_Y_VAL_KEY,
    S3_X_TEST_KEY,
    S3_Y_TEST_KEY,
    AWS_REGION,
)
from special_train.utils import get_aws_secret

aws_access_key = os.environ.get("AWS_ACCESS_KEY")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")


def launch_training(session, role_arn):
    sagemaker_session = sagemaker.Session(boto_session=session)

    estimator = TensorFlow(
        entry_point="train/train.py",
        source_dir="./special_train/",
        role=role_arn,
        instance_count=1,
        instance_type="ml.p2.xlarge",
        framework_version="2.6.0",
        output_path=f"s3://{S3_ETHEREUM_FORECAST_BUCKET}/models",
        py_version="py38",
        hyperparameters={
            "epochs": 10,
            "learning-rate": 0.0001,
            "lstm-units": 128,
            "batch-size": 32,
        },
        sagemaker_session=sagemaker_session,
    )

    try:
        estimator.fit(
            {
                "train_x": f"s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_X_TRAIN_KEY}",
                "train_y": f"s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_Y_TRAIN_KEY}",
                "valid_x": f"s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_X_VAL_KEY}",
                "valid_y": f"s3://{S3_ETHEREUM_FORECAST_BUCKET}/{S3_Y_VAL_KEY}",
            }
        )
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")


if __name__ == "__main__":

    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_access_key,
        region_name=AWS_REGION,
    )

    aws_secret_client = session.client(service_name="secretsmanager")

    role_arn = get_aws_secret(aws_secret_client, SECRET_ID, SECRET_SAGEMAKER_ARN_KEY)

    launch_training(session, role_arn)
