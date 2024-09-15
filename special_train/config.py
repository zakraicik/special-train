AWS_REGION = "us-east-1"
SECRET_ID = "ethereum-price-forecast"
SECRET_POLYGON_API_KEY = "polygonApi"
SECRET_SAGEMAKER_ARN_KEY = "sagemaker_arn"
S3_ETHEREUM_FORECAST_BUCKET = "ethereum-price-forcecast"
S3_ETHEREUM_CONSOLIDATED_RAW_PRICE_DATA_KEY = (
    "raw_data/ethereum_prices_consolidated.parquet"
)
S3_X_TRAIN_KEY = "modeling_data/X_train.npy"
S3_Y_TRAIN_KEY = "modeling_data/y_train.npy"
S3_X_VAL_KEY = "modeling_data/X_val.npy"
S3_Y_VAL_KEY = "modeling_data/y_val.npy"
S3_X_TEST_KEY = "modeling_data/X_test.npy"
S3_Y_TEST_KEY = "modeling_data/y_test.npy"
S3_FEATURE_SCLAER = "modeling_utils/feature_scaler.pkl"
S3_TARGET_SCALER = "modeling_utils/target_scaler.pkl"

N = 12

FEATURES = [
    "close",
    "volume",
    "transactions",
    "vwap",
]
