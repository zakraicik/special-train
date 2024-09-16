import pandas as pd
import json
import logging
import pandas as pd
import numpy as np
import pickle
import io
import os
import tempfile
import tensorflow as tf

from datetime import datetime
from io import BytesIO
from botocore.exceptions import ClientError
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def convert_millisecond_to_date(millisecond_timestamp):

    timestamp_in_seconds = millisecond_timestamp / 1000.0

    date_time = datetime.fromtimestamp(timestamp_in_seconds)

    return date_time


def convert_date_to_millisecond(date_time):

    timestamp_in_seconds = date_time.timestamp()

    millisecond_timestamp = int(timestamp_in_seconds * 1000)

    return millisecond_timestamp


def get_aws_secret(aws_secret_client, SecretId, secretName):
    try:
        get_secret_value_response = aws_secret_client.get_secret_value(
            SecretId=SecretId
        )
        secret = json.loads(get_secret_value_response["SecretString"])

        secret = secret[secretName]

        return secret

    except ClientError as e:
        raise e


def save_numpy_to_s3(aws_s3_client, np_array, s3_bucket, s3_key):
    """
    Save a numpy array to S3 as a .npy file.
    """

    buffer = BytesIO()

    np.save(buffer, np_array)
    buffer.seek(0)

    aws_s3_client.upload_fileobj(buffer, s3_bucket, s3_key)
    logger.info(f"Uploaded {s3_key} to S3 bucket {s3_bucket}")


def load_numpy_from_s3(s3_client, bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)

    file_stream = io.BytesIO(response["Body"].read())
    array = np.load(file_stream)
    return array


def parquet_to_s3(df, bucket, key, aws_s3_client):
    out_buffer = BytesIO()
    df.to_parquet(out_buffer, index=True)

    out_buffer.seek(0)

    aws_s3_client.upload_fileobj(
        Fileobj=out_buffer,
        Bucket=bucket,
        Key=key,
        ExtraArgs={"ContentType": "binary/octet-stream"},
    )


def save_object_to_s3(s3_client, obj, bucket, key):
    serialized_obj = pickle.dumps(obj)
    s3_client.put_object(Bucket=bucket, Key=key, Body=serialized_obj)


def load_object_from_s3(s3_client, bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    serialized_obj = response["Body"].read()
    obj = pickle.loads(serialized_obj)
    return obj


def save_model_to_s3(model, aws_s3_client, bucket):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_path = os.path.join(temp_dir, f"{timestamp}.keras")
        model.save(temp_path)

        s3_file = f"models/{timestamp}.keras"
        aws_s3_client.upload_file(temp_path, bucket, s3_file)


def _most_recent_model(aws_s3_client, bucket):

    response = aws_s3_client.list_objects_v2(Bucket=bucket, Prefix="models/")

    if "Contents" not in response:
        return None

    model_files = {}
    for obj in response["Contents"]:
        obj_key = obj["Key"]
        try:
            date_str = obj_key.split(".")[0].split("/")[-1]
            model_date = datetime.strptime(date_str, "%Y%m%d_%H%M%S")
            model_files[model_date] = obj_key
        except ValueError:
            continue

    if model_files:
        most_recent_date = max(model_files.keys())
        most_recent_file = model_files[most_recent_date]
        return most_recent_file.split("/")[-1]
    else:
        return None


def load_model_from_s3(aws_s3_client, bucket, model_name=None):

    if model_name is not None:
        pass
    else:
        model_name = _most_recent_model(aws_s3_client, bucket)

    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as temp_file:
        key = f"models/{model_name}"
        aws_s3_client.download_file(bucket, key, temp_file.name)
        model = tf.keras.models.load_model(temp_file.name)
    os.unlink(temp_file.name)
    return model


def load_raw_data(aws_s3_client, bucket, key):

    logger.info("Downloading raw data from S3...")

    response = aws_s3_client.get_object(Bucket=bucket, Key=key)
    parquet_buffer = BytesIO(response["Body"].read())

    df = pd.read_parquet(parquet_buffer)

    logger.info(f"Dataset Size: {df.shape}")

    logger.info("Reindexing... ")

    df.drop(columns=["otc"], inplace=True)
    df.dropna(inplace=True)

    df.set_index("timestamp", inplace=True)

    return df


def validate_timestamps(df):
    df.index = pd.to_datetime(df.index, unit="ms")

    df.sort_index(inplace=True)

    time_diffs = df.index.to_series().diff().dropna()

    assert (
        time_diffs == pd.Timedelta(minutes=5)
    ).all(), "Not all rows are 5 minutes apart"

    return df
