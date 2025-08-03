 #data_collection/mt5_data_loader.py
import sys
import os
import time
import pandas as pd
import MetaTrader5 as mt5
from pathlib import Path
from datetime import datetime, timedelta

# Add root directory to Python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from config import SYMBOLS, TIMEFRAME, HISTORICAL_BARS

# Timeframe mapping dictionary
TIMEFRAME_MAPPING = {
    '1min': mt5.TIMEFRAME_M1,
    '5min': mt5.TIMEFRAME_M5,
    '15min': mt5.TIMEFRAME_M15,
    '30min': mt5.TIMEFRAME_M30,
    '1h': mt5.TIMEFRAME_H1,
    '4h': mt5.TIMEFRAME_H4,
    '1d': mt5.TIMEFRAME_D1
}

def initialize_mt5(max_retries=5, retry_delay=10):
    """Initialize MT5 connection with retry logic"""
    for attempt in range(max_retries):
        if mt5.initialize():
            return True
        print(f"MT5 initialization failed (attempt {attempt+1}/{max_retries}), retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
    
    print(f"Failed to initialize MT5 after {max_retries} attempts")
    return False

def download_historical_data(symbol, timeframe_str, num_bars=HISTORICAL_BARS):
    """Download historical data for a symbol with proper timeframe handling"""
    # Get MT5 timeframe constant
    mt5_timeframe = TIMEFRAME_MAPPING.get(timeframe_str)
    if mt5_timeframe is None:
        print(f"Unsupported timeframe: {timeframe_str}")
        return None
    
    if not initialize_mt5():
        return None
    
    try:
        # Calculate time range for data request
        end_time = datetime.now()
        
        # Calculate start time based on timeframe
        if timeframe_str == '1min':
            start_time = end_time - timedelta(minutes=num_bars * 2)
        elif timeframe_str == '5min':
            start_time = end_time - timedelta(minutes=num_bars * 10)
        elif timeframe_str == '15min':
            start_time = end_time - timedelta(minutes=num_bars * 30)
        elif timeframe_str == '30min':
            start_time = end_time - timedelta(minutes=num_bars * 60)
        elif timeframe_str == '1h':
            start_time = end_time - timedelta(hours=num_bars * 2)
        elif timeframe_str == '4h':
            start_time = end_time - timedelta(hours=num_bars * 8)
        elif timeframe_str == '1d':
            start_time = end_time - timedelta(days=num_bars * 2)
        
        # Download data
        rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_time, end_time)
        
        if rates is None:
            print(f"No data retrieved for {symbol} ({timeframe_str})")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(rates)
        
        # Rename columns
        column_map = {
            'time': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume',
            'real_volume': 'volume'
        }
        
        # Apply column renaming
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        # Convert and set index
        df['date'] = pd.to_datetime(df['date'], unit='s')
        df.set_index('date', inplace=True)
        
        # Ensure we have the requested number of bars
        if len(df) < num_bars:
            print(f"Warning: Only {len(df)} bars retrieved for {symbol}, requested {num_bars}")
        
        return df[['open', 'high', 'low', 'close', 'volume']].tail(num_bars)
    
    except Exception as e:
        print(f"Error downloading data for {symbol}: {str(e)}")
        return None
    
    finally:
        mt5.shutdown()

def save_all_symbol_data(timeframe_str=TIMEFRAME):
    """Download and save data for all symbols"""
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Starting data download for {len(SYMBOLS)} symbols ({timeframe_str})...")
    
    for symbol in SYMBOLS:
        print(f"\nDownloading data for {symbol}...")
        
        # Download data
        data = download_historical_data(symbol, timeframe_str)
        
        if data is not None:
            # Save to CSV
            file_path = data_dir / f"{symbol}_{timeframe_str.replace('min', 'm')}.csv"
            data.to_csv(file_path)
            print(f"Saved {len(data)} bars to {file_path}")
        else:
            print(f"Failed to download data for {symbol}")
    
    print("\nData download completed!")

if __name__ == "__main__":
    # Download data for the timeframe specified in config
    save_all_symbol_data(TIMEFRAME)
    
    # Optionally download other timeframes
    # additional_timeframes = ['1h', '4h', '1d']
    # for tf in additional_timeframes:
    #     save_all_symbol_data(tf)