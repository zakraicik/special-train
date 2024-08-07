import pandas as pd
import numpy as np


def calculate_sma(df, feature, window):
    return df[feature].rolling(window=window).mean()


def apply_sma(df, settings, new_features):
    for feature, windows in settings["columns"].items():
        for window in windows:
            new_features[f"sma_{feature}_{window}"] = calculate_sma(df, feature, window)
    return new_features


def calculate_ema(df, feature, window):
    return df[feature].ewm(span=window, adjust=False).mean()


def apply_ema(df, settings, new_features):
    for feature, windows in settings["columns"].items():
        for window in windows:
            new_features[f"ema_{feature}_{window}"] = calculate_ema(df, feature, window)
    return new_features


def calculate_rsi(df, feature, window):
    delta = df[feature]
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def apply_rsi(df, settings, new_features):
    for feature, windows in settings["columns"].items():
        for window in windows:
            new_features[f"rsi_{feature}_{window}"] = calculate_rsi(df, feature, window)
    return new_features


def calculate_macd(df, feature, short_window, long_window):
    ema_short = df[feature].ewm(span=short_window, adjust=False).mean()
    ema_long = df[feature].ewm(span=long_window, adjust=False).mean()
    return ema_short - ema_long


def apply_macd(df, settings, new_features):
    for feature in settings["columns"]:
        for short_window, long_window in settings["windows"]:
            new_features[f"macd_{feature}_{short_window}_{long_window}"] = (
                calculate_macd(df, feature, short_window, long_window)
            )
    return new_features


def calculate_bollinger_bands(df, feature, window):
    sma = df[feature].rolling(window=window).mean()
    std = df[feature].rolling(window=window).std()
    bb_upper = sma + (std * 2)
    bb_lower = sma - (std * 2)
    return sma, bb_upper, bb_lower


def apply_bollinger_bands(df, settings, new_features):
    for feature, windows in settings["columns"].items():
        for window in windows:
            middle, upper, lower = calculate_bollinger_bands(df, feature, window)
            new_features[f"bb_middle_{feature}_{window}"] = middle
            new_features[f"bb_upper_{feature}_{window}"] = upper
            new_features[f"bb_lower_{feature}_{window}"] = lower
    return new_features


# ATR
def calculate_atr(df, window):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = high_low.combine(high_close, max).combine(low_close, max)
    return tr.rolling(window=window).mean()


def apply_atr(df, settings, new_features):
    for window in settings["windows"]:
        new_features[f"atr_{window}"] = calculate_atr(df, window)
    return new_features


# OBV
def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    obv_series = pd.Series(obv, index=df.index)
    return obv_series


def apply_obv(df, settings, new_features):
    new_features["obv"] = calculate_obv(df)
    return new_features


# Stochastic
def calculate_stochastic(df, window1, window2):
    low_min = df["low"].rolling(window=window1).min()
    high_max = df["high"].rolling(window=window1).max()
    percent_k = (df["close"] - low_min) * 100 / (high_max - low_min)
    percent_d = percent_k.rolling(window=window2).mean()
    return percent_k, percent_d


def apply_stochastic(df, settings, new_features):
    for window1, window2 in settings["windows"]:
        percent_k, percent_d = calculate_stochastic(df, window1, window2)
        new_features[f"percent_k_{window1}_{window2}"] = percent_k
        new_features[f"percent_d_{window1}_{window2}"] = percent_d
    return new_features


# CCI
def calculate_cci(df, window):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.fabs(x - x.mean()).mean())
    return (tp - sma_tp) / (0.015 * mad)


def apply_cci(df, settings, new_features):
    for window in settings["windows"]:
        new_features[f"cci_{window}"] = calculate_cci(df, window)
    return new_features


# CMF
def calculate_cmf(df, window):
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"]
    )
    mfv = mfm * df["volume"]
    return mfv.rolling(window=window).sum() / df["volume"].rolling(window=window).sum()


def apply_cmf(df, settings, new_features):
    for window in settings["windows"]:
        new_features[f"cmf_{window}"] = calculate_cmf(df, window)
    return new_features


# Pivot Points
def calculate_pivot_points(df):
    pivot = (df["high"] + df["low"] + df["close"]) / 3
    r1 = 2 * pivot - df["low"]
    s1 = 2 * pivot - df["high"]
    r2 = pivot + (df["high"] - df["low"])
    s2 = pivot - (df["high"] - df["low"])
    r3 = pivot + 2 * (df["high"] - df["low"])
    s3 = pivot - 2 * (df["high"] - df["low"])
    return pivot, r1, s1, r2, s2, r3, s3


def apply_pivot_points(df, settings, new_features):
    pivot, r1, s1, r2, s2, r3, s3 = calculate_pivot_points(df)
    new_features["pivot"] = pivot
    new_features["r1"] = r1
    new_features["s1"] = s1
    new_features["r2"] = r2
    new_features["s2"] = s2
    new_features["r3"] = r3
    new_features["s3"] = s3
    return new_features


feature_functions = {
    "sma": apply_sma,
    "ema": apply_ema,
    "rsi": apply_rsi,
    "macd": apply_macd,
    "bollinger_bands": apply_bollinger_bands,
    "atr": apply_atr,
    "obv": apply_obv,
    "stochastic": apply_stochastic,
    "cci": apply_cci,
    "cmf": apply_cmf,
    "pivot_points": apply_pivot_points,
}


def generate_training_features(df, config):
    new_features = {}
    for feature, settings in config.items():
        if feature in feature_functions:
            new_features = feature_functions[feature](df, settings, new_features)
    new_features_df = pd.DataFrame(new_features)
    df = pd.concat([df, new_features_df], axis=1)
    return df
