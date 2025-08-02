# feature_engineering/advanced_features.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import requests
import os
from .macro_data_manager import MacroDataManager
from sentiment.sentiment_downloader import SentimentDownloader  # Import sentiment downloadera

class AdvancedFeatureEngineer:
    def __init__(self, symbol, mode='training'):
        """
        Initialize feature engineer
        :param symbol: Trading symbol (e.g., BTCUSD)
        :param mode: 'training' or 'prediction' - controls data handling
        """
        self.symbol = symbol
        self.mode = mode
        self.us_holidays = holidays.US()
        self.data_manager = MacroDataManager()
        self.sentiment_downloader = SentimentDownloader()  # Initialize sentiment downloader
        
        # Mapovanie typov aktív pre sentiment
        self.asset_type_map = {
            'BTCUSD_ecn': 'crypto',
            'ETHUSD_ecn': 'crypto',
            'EURUSD_ecn': 'forex',
            'XAUUSD_ecn': 'commodity',
            'USOIL.fut': 'commodity',
            'DAX_ecn': 'stock',
            'SP_ecn': 'stock',
            'NSDQ_ecn': 'stock'
        }
        
        # Handle macro data based on mode
        self._handle_macro_data()
    
    def _get_asset_type(self):
        """Získaj typ aktíva pre sentiment"""
        return self.asset_type_map.get(self.symbol, 'stock')
    
    def _handle_macro_data(self):
        """Manage macro data based on current mode"""
        if self.mode == 'training':
            # Download all historical data for training
            self.data_manager.download_all_historical()
        elif self.mode == 'prediction':
            # Update to latest data for prediction
            self.data_manager.update_all()
    
    def add_seasonal_features(self, df):
        """Add seasonal and calendar features"""
        # Day of week (0=Monday, 6=Sunday)
        df['day_of_week'] = df.index.dayofweek
        
        # Month of year
        df['month'] = df.index.month
        
        # Quarter of year
        df['quarter'] = df.index.quarter
        
        # Hour of day (for intraday data)
        df['hour'] = df.index.hour
        
        # Is US holiday?
        df['is_holiday'] = df.index.date.astype('datetime64').isin(self.us_holidays)
        
        # Weekend flag (Friday after close to Sunday)
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Month start/end effects
        df['is_month_start'] = (df.index.day == 1).astype(int)
        df['is_month_end'] = (df.index.is_month_end).astype(int)
        
        # Quarter start/end effects
        df['is_quarter_start'] = ((df.index.month - 1) % 3 == 0) & (df.index.day == 1)
        df['is_quarter_end'] = (df.index.is_quarter_end).astype(int)
        
        return df
    
    def add_technical_indicators(self, df):
        """Add advanced technical indicators"""
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = sma20 + (std20 * 2)
        df['bollinger_lower'] = sma20 - (std20 * 2)
        df['bollinger_percent'] = (df['close'] - df['bollinger_lower']) / (df['bollinger_upper'] - df['bollinger_lower'])
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.max(np.array([high_low, high_close, low_close]), axis=0)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume-based features
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_roc'] = df['volume'].pct_change(periods=5)
        
        # Price volatility
        df['volatility'] = df['close'].rolling(window=20).std() * np.sqrt(20)
        
        return df
    
    def add_macro_features(self, df):
        """Add macroeconomic indicators relevant to symbol"""
        # Use data manager to get macro data instead of direct API calls
        if 'USOIL' in self.symbol:
            try:
                oil_data = self.data_manager.get_oil_inventory()
                df = df.join(oil_data, how='left')
                df['oil_inventory'] = df['oil_inventory'].ffill().bfill()
            except Exception as e:
                print(f"Error loading oil data: {str(e)}")
                df['oil_inventory'] = 0
        
        if 'XAUUSD' in self.symbol:
            try:
                gold_data = self.data_manager.get_gold_reserves()
                df = df.join(gold_data, how='left')
                df['gold_reserves'] = df['gold_reserves'].ffill().bfill()
            except Exception as e:
                print(f"Error loading gold data: {str(e)}")
                df['gold_reserves'] = 0
        
        # For stock indices
        if any(x in self.symbol for x in ['NSDQ', 'SP', 'DAX']):
            try:
                vix_data = self.data_manager.get_vix()
                df = df.join(vix_data, how='left')
                df['VIX'] = df['VIX'].ffill().bfill()
            except Exception as e:
                print(f"Error loading VIX data: {str(e)}")
                df['VIX'] = 0
                
        return df
    
    def add_geopolitical_risk(self, df):
        """Add geopolitical risk indicator"""
        try:
            gpr_data = self.data_manager.get_geopolitical_risk()
            df = df.join(gpr_data, how='left')
            df['gpr_index'] = df['gpr_index'].ffill().bfill()
        except Exception as e:
            print(f"Error loading geopolitical data: {str(e)}")
            df['gpr_index'] = 0
            
        return df
    
    def add_sentiment_features(self, df):
        """Pridaj sentimentové features k OHLCV dátam"""
        # Získaj typ aktíva
        asset_type = self._get_asset_type()
        
        # Vytvor stĺpce pre sentiment
        df['market_sentiment'] = 0.0
        df['news_sentiment'] = 0.0
        
        # Prejdi každý deň v dátach
        unique_dates = df.index.normalize().unique()
        
        for date in unique_dates:
            # Získaj sentiment pre daný deň
            market_sentiment = self.sentiment_downloader.get_market_sentiment(
                self.symbol, asset_type, 
                from_time=date, 
                to_time=date + timedelta(days=1)
            )
            
            news_sentiment = self.sentiment_downloader.get_news_sentiment(
                self.symbol, 
                from_time=date - timedelta(days=1),  # Správy z predchádzajúceho dňa
                to_time=date
            )
            
            # Pridaj sentiment k dátam
            if market_sentiment:
                # Normalizuj sentiment na rozsah -1 až 1
                if 'sentiment' in market_sentiment:
                    df.loc[df.index.normalize() == date, 'market_sentiment'] = market_sentiment['sentiment']
            
            if news_sentiment:
                df.loc[df.index.normalize() == date, 'news_sentiment'] = news_sentiment['news_sentiment']
        
        # Dopln chýbajúce hodnoty
        df['market_sentiment'] = df['market_sentiment'].ffill().bfill()
        df['news_sentiment'] = df['news_sentiment'].ffill().bfill()
        
        return df
    
    def add_crypto_specific_features(self, df):
        """Add crypto-specific features"""
        if 'BTC' in self.symbol:
            try:
                dominance_data = self.data_manager.get_btc_dominance()
                df = df.join(dominance_data, how='left')
                df['btc_dominance'] = df['btc_dominance'].ffill().bfill()
                
                # Add other crypto features
                df = self._add_crypto_sentiment(df)
            except Exception as e:
                print(f"Error loading crypto data: {str(e)}")
                df['btc_dominance'] = 0
                
        return df
    
    def _add_crypto_sentiment(self, df):
        """Add cryptocurrency sentiment features"""
        # Implement sentiment analysis here
        return df
    
    def add_all_features(self, df):
        """Add all advanced features"""
        # Pridaj základné technické indikátory
        df = self.add_technical_indicators(df)
        
        # Pridaj sezónne a kalendárne features
        df = self.add_seasonal_features(df)
        
        # Pridaj makroekonomické indikátory
        df = self.add_macro_features(df)
        
        # Pridaj geopolitické riziká
        df = self.add_geopolitical_risk(df)
        
        # Pridaj sentimentové features
        df = self.add_sentiment_features(df)
        
        # Pridaj špecifické features pre kryptomeny
        if 'BTC' in self.symbol:
            df = self.add_crypto_specific_features(df)
            
        # Odstráň dočasné stĺpce a NaN hodnoty
        df = df.dropna()
        df = df.drop(columns=['is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end'], errors='ignore')
        
        return df