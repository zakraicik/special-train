import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def convert_millisecond_to_date(millisecond_timestamp):

    timestamp_in_seconds = millisecond_timestamp / 1000.0

    date_time = datetime.fromtimestamp(timestamp_in_seconds)

    return date_time


def convert_date_to_millisecond(date_time):

    timestamp_in_seconds = date_time.timestamp()

    millisecond_timestamp = int(timestamp_in_seconds * 1000)

    return millisecond_timestamp


def validate_timestamps(df):
    df.index = pd.to_datetime(df.index, unit="ms")

    df.sort_index(inplace=True)

    time_diffs = df.index.to_series().diff().dropna()

    assert (
        time_diffs == pd.Timedelta(minutes=5)
    ).all(), "Not all rows are 5 minutes apart"

    return df
