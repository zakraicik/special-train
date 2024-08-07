import sagemaker
from sagemaker.tensorflow import TensorFlow

s3_input_train = "s3://your-bucket/path-to-train-data"
s3_input_test = "s3://your-bucket/path-to-test-data"
s3_output_path = "s3://your-bucket/path-to-model-output"

tf_estimator = TensorFlow(
    entry_point="train.py",
    role="your-sagemaker-role",
    instance_count=1,
    instance_type="ml.p2.xlarge",  # or another instance type with GPU
    framework_version="2.6.0",
    py_version="py37",
    hyperparameters={"epochs": 10, "batch-size": 64, "learning-rate": 0.001},
    output_path=s3_output_path,
    sagemaker_session=sagemaker.Session(),
)

tf_estimator.fit({"train": s3_input_train, "test": s3_input_test})
