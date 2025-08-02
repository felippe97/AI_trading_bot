import pandas as pd
import numpy as np
import pandas_ta as ta  # Toto je správny import

def compute_intraday_features(df, timeframe='5min'):
    # Konvercia časového rámca na minúty
    if 'min' in timeframe:
        timeframe_minutes = int(timeframe.replace('min', ''))
    else:
        timeframe_minutes = 1  # fallback
   
    
    # Dynamické parametre podľa časového rámca
    base_period = max(1, int(timeframe_minutes / 5))
    ema_period = base_period * 50
    rsi_period = base_period * 14
    atr_period = base_period * 14
    bb_period = base_period * 20
    macd_fast = max(12, int(12 * base_period))
    macd_slow = max(26, int(26 * base_period))
    macd_signal = max(9, int(9 * base_period))
    
    # Výpočet indikátorov pomocou pandas_ta
    df['ema'] = ta.ema(df['close'], length=ema_period)
    df['rsi'] = ta.rsi(df['close'], length=rsi_period)
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
    
    # Bollinger Bands
    bb = ta.bbands(df['close'], length=bb_period)
    df = pd.concat([df, bb], axis=1)
    df.rename(columns={
        f'BBU_{bb_period}_2.0': 'bb_upper',
        f'BBM_{bb_period}_2.0': 'bb_mid',
        f'BBL_{bb_period}_2.0': 'bb_lower'
    }, inplace=True)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    # MACD
    macd = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df = pd.concat([df, macd], axis=1)
    df.rename(columns={
        f'MACD_{macd_fast}_{macd_slow}_{macd_signal}': 'macd',
        f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}': 'macd_signal',
        f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}': 'macd_hist'
    }, inplace=True)
    
    # Objemové indikátory
    df['vol_ma'] = df['volume'].rolling(window=20).mean()
    df['vol_z'] = (df['volume'] - df['vol_ma']) / df['volume'].rolling(window=20).std()
    
    # Časové features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_part'] = pd.cut(df.index.hour, 
                            bins=[0, 8, 12, 16, 24], 
                            labels=['night', 'morning', 'midday', 'afternoon'],
                            include_lowest=True)
    
    # Ceny a výnosy
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['range_pct'] = (df['high'] - df['low']) / df['open']
    
    # Vymazanie NaN hodnôt
    df.dropna(inplace=True)
    return df