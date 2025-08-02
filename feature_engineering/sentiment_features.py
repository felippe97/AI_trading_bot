# feature_engineering/sentiment_features.py
import pandas as pd
from datetime import datetime, timedelta
from sentiment.sentiment_downloader import SentimentDownloader

class SentimentFeatureEngineer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.downloader = SentimentDownloader()
        
        # Mapovanie typov aktív
        self.asset_type_map = {
            'BTCUSD_ecn': 'crypto',
            'ETHUSD_ecn': 'crypto',
            'EURUSD_ecn': 'forex',
            'XAUUSD_ecn': 'forex',
            'USOIL.fut': 'commodity',
            'DAX_ecn': 'stock',
            'SP_ecn': 'stock',
            'NSDQ_ecn': 'stock'
        }
    
    def add_sentiment_features(self, df):
        """Pridaj sentimentové features k OHLCV dátam"""
        # Získaj typ aktíva
        asset_type = self.asset_type_map.get(self.symbol, 'stock')
        
        # Vytvor stĺpce pre sentiment
        df['market_sentiment'] = 0.0
        df['news_sentiment'] = 0.0
        
        # Prejdi každý deň v dátach
        unique_dates = df.index.normalize().unique()
        
        for date in unique_dates:
            # Získaj sentiment pre daný deň
            market_sentiment = self.downloader.get_market_sentiment(
                self.symbol, asset_type, 
                from_time=date, 
                to_time=date + timedelta(days=1)
            )
            
            news_sentiment = self.downloader.get_news_sentiment(
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
        
        return df