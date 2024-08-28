AWS_REGION = "us-east-1"
SECRET_ID = "ethereum-price-forecast"
SECRET_POLYGON_API_KEY = "polygonApi"
SECRET_SAGEMAKER_ARN_KEY = "sagemaker_arn"
S3_ETHEREUM_FORECAST_BUCKET = "ethereum-price-forcecast"
S3_ETHEREUM_CONSOLIDATED_RAW_PRICE_DATA_KEY = (
    "raw_data/ethereum_prices_consolidated.parquet"
)
S3_TRAIN_KEY = "modeling_data/train.parquet"
S3_VAL_KEY = "modeling_data/val.parquet"
S3_TEST_KEY = "modeling_data/test.parquet"

RAW_FEATURES = ["open", "high", "low", "close", "volume", "transactions"]

FEATURE_CONFIG = {
    "sma": {
        "columns": {
            "open": [5, 10, 30, 60, 120, 288, 864],
            "high": [5, 10, 30, 60, 120, 288, 864],
            "low": [5, 10, 30, 60, 120, 288, 864],
            "close": [5, 10, 30, 60, 120, 288, 864],
            "volume": [5, 10, 30, 60, 120, 288, 864],
            "vwap": [5, 10, 30, 60, 120, 288, 864],
            "transactions": [5, 10, 30, 60, 120, 288, 864],
        }
    },
    "ema": {
        "columns": {
            "open": [5, 10, 30, 60, 120, 288, 864],
            "high": [5, 10, 30, 60, 120, 288, 864],
            "low": [5, 10, 30, 60, 120, 288, 864],
            "close": [5, 10, 30, 60, 120, 288, 864],
            "volume": [5, 10, 30, 60, 120, 288, 864],
            "vwap": [5, 10, 30, 60, 120, 288, 864],
            "transactions": [5, 10, 30, 60, 120, 288, 864],
        }
    },
    "rsi": {
        "columns": {
            "open": [5, 10, 30, 60, 120, 288, 864],
            "high": [5, 10, 30, 60, 120, 288, 864],
            "low": [5, 10, 30, 60, 120, 288, 864],
            "close": [5, 10, 30, 60, 120, 288, 864],
            "volume": [5, 10, 30, 60, 120, 288, 864],
            "vwap": [5, 10, 30, 60, 120, 288, 864],
            "transactions": [5, 10, 30, 60, 120, 288, 864],
        }
    },
    "macd": {
        "columns": ["open", "high", "low", "close", "volume", "vwap", "transactions"],
        "windows": [(12, 26), (30, 60), (60, 120)],  # short, medium, long-term pairs
    },
    "bollinger_bands": {
        "columns": {
            "open": [20, 30, 60, 120, 288, 864],
            "high": [20, 30, 60, 120, 288, 864],
            "low": [20, 30, 60, 120, 288, 864],
            "close": [20, 30, 60, 120, 288, 864],
            "volume": [20, 30, 60, 120, 288, 864],
            "vwap": [20, 30, 60, 120, 288, 864],
            "transactions": [20, 30, 60, 120, 288, 864],
        }
    },
    "atr": {"windows": [14, 30, 60, 120, 288, 864]},
    "obv": {"apply": True},
    "stochastic": {"windows": [(14, 3), (30, 10), (60, 20)]},
    "cci": {"windows": [20, 30, 60, 120, 288, 864]},
    "cmf": {"windows": [20, 30, 60, 120, 288, 864]},
    "pivot_points": {"apply": True},
}

LAG_PERIODS = [1, 2, 3, 5, 10, 30, 60, 120, 288, 864]

RAW_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "transactions"]

TARGET = "next_period_close_change"
TOP_N_FEATURES = 100

SEQUENCE_LENGTH = 12
