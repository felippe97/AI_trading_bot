# data_downloader.py
import os
import pandas as pd
import numpy as np
import time
import requests
from datetime import datetime, timedelta
import logging
from config import project_root, SYMBOLS, TIMEFRAME, HISTORICAL_BARS

logger = logging.getLogger(__name__)

class DataDownloader:
    API_SOURCES = {
        'crypto': 'binance',
        'forex': 'alpha_vantage',
        'commodities': 'alpha_vantage',
        'indices': 'alpha_vantage'
    }
    
    TIMEFRAME_MAP = {
        '5min': {
            'binance': '5m',
            'alpha_vantage': '5min'
        },
        '15min': {
            'binance': '15m',
            'alpha_vantage': '15min'
        },
        '1h': {
            'binance': '1h',
            'alpha_vantage': '60min'
        },
        '1d': {
            'binance': '1d',
            'alpha_vantage': 'daily'
        }
    }
    
    SYMBOL_MAP = {
        'BTCUSD_ecn': ('BTCUSDT', 'crypto'),
        'ETHUSD_ecn': ('ETHUSDT', 'crypto'),
        'EURUSD_ecn': ('EURUSD', 'forex'),
        'XAUUSD_ecn': ('GOLD', 'commodities'),
        'USOIL.fut': ('WTI', 'commodities'),
        'DAX_ecn': ('DAX', 'indices'),
        'SP_ecn': ('SPX', 'indices'),
        'NSDQ_ecn': ('NASDAQ', 'indices')
    }
    
    def __init__(self):
        self.api_keys = {
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_KEY', 'YOUR_DEFAULT_API_KEY')
        }
        
    def download_data(self, symbol, timeframe, bars=HISTORICAL_BARS):
        """Download historical data for given symbol and timeframe"""
        if symbol not in self.SYMBOL_MAP:
            logger.error(f"No mapping for symbol: {symbol}")
            return None
            
        api_symbol, asset_type = self.SYMBOL_MAP[symbol]
        api_source = self.API_SOURCES[asset_type]
        
        logger.info(f"Downloading {bars} bars of {symbol} ({timeframe}) from {api_source}")
        
        if api_source == 'binance':
            return self._download_binance(api_symbol, timeframe, bars)
        elif api_source == 'alpha_vantage':
            return self._download_alpha_vantage(api_symbol, timeframe, bars, asset_type)
        else:
            logger.error(f"Unsupported API source: {api_source}")
            return None
    
    def _download_binance(self, symbol, timeframe, bars):
        """Download data from Binance public API"""
        interval = self.TIMEFRAME_MAP[timeframe]['binance']
        url = f"https://api.binance.com/api/v3/klines"
        
        # Calculate start time based on number of bars
        end_time = int(datetime.now().timestamp() * 1000)
        
        # Binance limits to 1000 records per request
        all_data = []
        while bars > 0:
            limit = min(bars, 1000)
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit,
                'endTime': end_time
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                    
                all_data = data + all_data
                bars -= len(data)
                
                # Move end_time to the earliest record
                end_time = data[0][0] - 1
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error downloading from Binance: {str(e)}")
                break
        
        return self._parse_binance_data(all_data)
    
    def _parse_binance_data(self, data):
        """Parse Binance API response into DataFrame"""
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ]
        
        df = pd.DataFrame(data, columns=columns, dtype=float)
        df['date'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('date', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        return df
    
    def _download_alpha_vantage(self, symbol, timeframe, bars, asset_type):
        """Download data from Alpha Vantage API"""
        interval = self.TIMEFRAME_MAP[timeframe]['alpha_vantage']
        
        # Alpha Vantage function mapping
        function_map = {
            'forex': 'FX_INTRADAY',
            'commodities': 'CURRENCY_INTRADAY',
            'indices': 'TIME_SERIES_INTRADAY'
        }
        
        function = function_map.get(asset_type, 'TIME_SERIES_INTRADAY')
        
        url = f"https://www.alphavantage.co/query"
        params = {
            'function': function,
            'symbol': symbol,
            'interval': interval,
            'apikey': self.api_keys['alpha_vantage'],
            'outputsize': 'full',
            'datatype': 'json'
        }
        
        # Adjust params for different asset types
        if asset_type == 'forex':
            params['from_symbol'] = symbol[:3]
            params['to_symbol'] = symbol[3:]
            del params['symbol']
        elif asset_type == 'commodities':
            params['from_currency'] = symbol
            params['to_currency'] = 'USD'
            del params['symbol']
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Find the actual data in the response
            for key in data.keys():
                if "Time Series" in key:
                    data_key = key
                    break
            else:
                logger.error("No time series data found in Alpha Vantage response")
                return None
                
            df = pd.DataFrame.from_dict(data[data_key], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            })[['open', 'high', 'low', 'close', 'volume']]
            df = df.sort_index()
            df = df.astype(float)
            
            # Limit to requested number of bars
            return df.tail(bars)
            
        except Exception as e:
            logger.error(f"Error downloading from Alpha Vantage: {str(e)}")
            return None

    def save_all_data(self, timeframe=TIMEFRAME, bars=HISTORICAL_BARS):
        """Download and save data for all symbols"""
        data_dir = os.path.join(project_root, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        for symbol in SYMBOLS:
            file_path = os.path.join(data_dir, f"{symbol}_{timeframe}.csv")
            
            # Skip if file already exists
            if os.path.exists(file_path):
                logger.info(f"Data file exists: {file_path}")
                continue
                
            df = self.download_data(symbol, timeframe, bars)
            
            if df is not None and not df.empty:
                df.to_csv(file_path)
                logger.info(f"Saved {len(df)} records to {file_path}")
            else:
                logger.warning(f"No data downloaded for {symbol}")
                
        return True

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    downloader = DataDownloader()
    downloader.save_all_data()