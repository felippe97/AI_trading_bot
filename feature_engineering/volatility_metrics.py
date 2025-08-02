# feature_engineering/volatility_metrics.py
import pandas as pd
import numpy as np
import pandas_ta as ta
def calculate_intraday_volatility(df, window=20):
    """Vypočíta intradennú volatilitu"""
    range_pct = (df['high'] - df['low']) / df['open']
    return range_pct.rolling(window=window).mean()

def calculate_price_swings(df, threshold=0.001):
    """Identifikuje významné cenové pohyby"""
    price_changes = df['close'].pct_change()
    return (np.abs(price_changes) > threshold).astype(int)