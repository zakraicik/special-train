import os
import boto3
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import t
from datetime import datetime
from special_train.config import (
    AWS_REGION,
    S3_ETHEREUM_FORECAST_BUCKET,
    S3_X_TRAIN_KEY,
    S3_Y_TRAIN_KEY,
    S3_X_VAL_KEY,
    S3_Y_VAL_KEY,
    S3_X_TEST_KEY,
    S3_Y_TEST_KEY,
    N,
)
from special_train.utils import (
    load_model_from_s3,
    load_numpy_from_s3,
    load_object_from_s3,
)

aws_access_key = os.environ.get("AWS_ACCESS_KEY")
aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

session = boto3.Session(
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_access_key,
    region_name=AWS_REGION,
)

aws_secret_client = session.client(service_name="secretsmanager")
aws_s3_client = session.client(service_name="s3")


def plot_actual_vs_expected(
    targets, predictions, N, filename="special_train/eval/actual_vs_expected.png"
):

    plt.figure(figsize=(12, 6))

    plt.plot(targets, label="Actual Close Price", linestyle="-", alpha=0.7)

    forecast_steps = N
    for i in range(0, len(predictions) - forecast_steps + 1, forecast_steps):
        plt.plot(
            range(i, i + forecast_steps),
            predictions[i : i + forecast_steps],
            linestyle="--",
            alpha=0.6,
            color="red",
        )

    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.title("Actual vs. Predicted Close Price (Multistep)")

    plt.savefig(os.path.join(os.getcwd(), filename), bbox_inches="tight")
    plt.close()


def plot_error_scatter(
    targets, predictions, filename="special_train/eval/error_scatter.png"
):

    plt.figure(figsize=(8, 8))

    plt.scatter(predictions, targets, alpha=0.5, label="Predicted vs. Actual")

    min_val = min(np.min(predictions), np.min(targets))
    max_val = max(np.max(predictions), np.max(targets))

    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Match")

    plt.xlabel("Predicted Close Price")
    plt.ylabel("Actual Close Price")
    plt.title("Predicted vs. Actual Close Price")
    plt.legend()

    plt.savefig(os.path.join(os.getcwd(), filename), bbox_inches="tight")
    plt.close()


def plot_confidence_by_timestep(
    targets,
    predictions,
    confidence_level=0.95,
    filename="special_train/eval/confidence_by_timestep.png",
):

    num_samples, N = predictions.shape

    absolute_errors = np.abs(targets - predictions)

    mae_by_step = np.mean(absolute_errors, axis=0)

    std_error = np.std(absolute_errors, axis=0, ddof=1) / np.sqrt(num_samples)

    degrees_freedom = num_samples - 1

    t_crit = np.abs(t.ppf((1 - confidence_level) / 2, degrees_freedom))

    margin_of_error = t_crit * std_error

    ci_lower = mae_by_step - margin_of_error
    ci_upper = mae_by_step + margin_of_error

    plt.figure(figsize=(12, 6))
    timesteps = np.arange(1, N + 1)

    plt.plot(
        timesteps,
        mae_by_step,
        marker="o",
        linestyle="-",
        color="b",
        label="Mean Absolute Error",
    )

    plt.fill_between(
        timesteps,
        ci_lower,
        ci_upper,
        color="b",
        alpha=0.2,
        label=f"{int(confidence_level * 100)}% Confidence Interval",
    )

    plt.xlabel("Time Step Ahead")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.title("Mean Absolute Error by Time Step with Confidence Intervals")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(os.getcwd(), filename), bbox_inches="tight")
    plt.close()


def plot_error_distribution_by_timestep(
    targets,
    predictions,
    filename="special_train/eval/error_distribution_by_timestep.png",
):

    num_samples, N = predictions.shape

    errors = targets - predictions

    plt.figure(figsize=(12, 8))

    for step in range(N):
        plt.subplot(N // 2, 2, step + 1)
        plt.hist(
            errors[:, step],
            bins=30,
            edgecolor="k",
            alpha=0.7,
        )
        plt.title(f"Error Distribution for Step {step + 1}")
        plt.xlabel("Error")
        plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(os.path.join(os.getcwd(), filename), bbox_inches="tight")
    plt.close()


if __name__ == "__main__":

    X_test = load_numpy_from_s3(
        aws_s3_client, S3_ETHEREUM_FORECAST_BUCKET, S3_X_TEST_KEY
    )

    y_test = load_numpy_from_s3(
        aws_s3_client, S3_ETHEREUM_FORECAST_BUCKET, S3_Y_TEST_KEY
    )

    model = load_model_from_s3(aws_s3_client, S3_ETHEREUM_FORECAST_BUCKET)

    target_scaler = load_object_from_s3(
        aws_s3_client, S3_ETHEREUM_FORECAST_BUCKET, "modeling_utils/target_scaler.pkl"
    )

    inversed_targets = target_scaler.inverse_transform(y_test)

    predicted_close = model.predict(X_test)
    inversed_predicted_close = target_scaler.inverse_transform(predicted_close)

    plot_actual_vs_expected(inversed_targets, inversed_predicted_close, N)

    plot_error_scatter(inversed_targets, inversed_predicted_close)

    plot_confidence_by_timestep(inversed_targets, inversed_predicted_close)

    plot_error_distribution_by_timestep(inversed_targets, inversed_predicted_close)
