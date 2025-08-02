# sentiment/sentiment_downloader.py
import sys
import os
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
import json

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)
# Get the root directory (two levels up from the script)
project_root = os.path.dirname(os.path.dirname(current_script_path))
# Add root to Python path
sys.path.insert(0, project_root)

# Now import config
from config import FINNHUB_API_KEY, CRYPTOCOMPARE_API_KEY, NEWSAPI_API_KEY

# Nastavte podrobné logovanie
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sentiment_downloader.log')
    ]
)
logger = logging.getLogger(__name__)

class SentimentDownloader:
    def get_market_sentiment(self, symbol, asset_type, from_time=None, to_time=None):
        """Získaj sentiment pre daný symbol a typ aktíva"""
        if not from_time:
            from_time = datetime.now() - timedelta(days=30)
        if not to_time:
            to_time = datetime.now()
            
        if asset_type == 'crypto':
            return self._get_crypto_sentiment(symbol, from_time, to_time)
        elif asset_type == 'forex':
            return self._get_forex_news_sentiment(symbol)
        else:  # stocks, indices
            return self._get_stock_sentiment(symbol, from_time, to_time)
    
    def _get_crypto_sentiment(self, symbol, from_time, to_time):
        """Získaj sentiment pre kryptomeny"""
        url = "https://min-api.cryptocompare.com/data/tradingsignals/intotheblock/latest"
        params = {
            'api_key': CRYPTOCOMPARE_API_KEY,
            'fsym': symbol.replace('USD_ecn', '')  # BTCUSD_ecn -> BTC
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Nový spôsob spracovania odpovede
            if 'Data' not in data:
                logger.warning(f"Crypto response missing data: {json.dumps(data, indent=2)}")
                return None
                
            indicators = data['Data']
            bullish_count = 0
            bearish_count = 0
            total_indicators = 0
            
            # Prejdi všetky indikátory v odpovedi
            for key, indicator in indicators.items():
                if key in ['id', 'time', 'symbol', 'partner_symbol']:
                    continue
                    
                sentiment = indicator.get('sentiment', '').lower()
                if sentiment == 'bullish':
                    bullish_count += 1
                elif sentiment == 'bearish':
                    bearish_count += 1
                    
                total_indicators += 1
            
            if total_indicators == 0:
                return None
                
            # Vypočítaj celkový sentiment
            sentiment_score = (bullish_count - bearish_count) / total_indicators
            return {
                'sentiment': sentiment_score,
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'total_indicators': total_indicators
            }
        except Exception as e:
            logger.error(f"Crypto sentiment error: {str(e)}")
            return None
    
    def _get_forex_news_sentiment(self, symbol):
        """Získaj sentiment pre forex pomocou novín"""
        symbol_map = {
            'EURUSD': 'euro OR eur usd OR ecb',
            'GBPUSD': 'pound OR gbp usd OR bank of england',
            'USDJPY': 'yen OR usd jpy OR bank of japan'
        }
        
        clean_symbol = symbol.replace('_ecn', '')
        query = symbol_map.get(clean_symbol, clean_symbol)
        return self.get_news_sentiment(query)
    
    def _get_stock_sentiment(self, symbol, from_time, to_time):
        """Získaj sentiment pre akcie a indexy pomocou NewsAPI"""
        clean_symbol = symbol.split('_')[0]  # DAX_ecn -> DAX
        
        # Mapovanie symbolov na vyhľadávacie výrazy
        symbol_map = {
            'DAX': 'germany stock OR dax index',
            'SP': 's&p 500 OR spx',
            'NSDQ': 'nasdaq OR ndx'
        }
        
        query = symbol_map.get(clean_symbol, clean_symbol)
        return self.get_news_sentiment(query, from_time, to_time)
    
    def get_news_sentiment(self, query, from_time=None, to_time=None):
        """Získaj sentiment z novinových článkov"""
        if not from_time:
            from_time = datetime.now() - timedelta(days=7)
        if not to_time:
            to_time = datetime.now()
            
        url = "https://newsapi.org/v2/everything"
        params = {
            'apiKey': NEWSAPI_API_KEY,
            'q': query,
            'from': from_time.strftime('%Y-%m-%d'),
            'to': to_time.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'relevancy',
            'pageSize': 50
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Kontrola článkov v odpovedi
            articles = data.get('articles', [])
            if not articles:
                logger.warning(f"No articles found for query: {query}")
                return {'news_sentiment': 0, 'article_count': 0}
            
            # Vylepšená analýza sentimentu
            sentiment_scores = []
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                text = f"{title}. {description}"
                score = self._advanced_sentiment_analysis(text)
                sentiment_scores.append(score)
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            return {
                'news_sentiment': avg_sentiment,
                'article_count': len(articles)
            }
        except Exception as e:
            logger.error(f"News sentiment error: {str(e)}")
            return None
    
    def _advanced_sentiment_analysis(self, text):
        """Vylepšená heuristika pre sentiment"""
        if not text:
            return 0
            
        positive_words = ['up', 'rise', 'gain', 'bull', 'positive', 'strong', 'buy', 'rally', 'growth', 
                         'increase', 'high', 'recovery', 'profit', 'surge', 'boost', 'outperform']
        negative_words = ['down', 'fall', 'drop', 'bear', 'negative', 'weak', 'sell', 'crash', 'decline',
                         'decrease', 'low', 'loss', 'plunge', 'dip', 'underperform', 'crisis']
        
        text_lower = text.lower()
        
        # Vážený sentiment
        positive_score = 0
        negative_score = 0
        
        # Pozitívne slová
        for word in positive_words:
            if word in text_lower:
                # Dôležitosť podľa početnosti
                positive_score += text_lower.count(word) * 1.5
                
        # Negatívne slová
        for word in negative_words:
            if word in text_lower:
                negative_score += text_lower.count(word) * 1.5
                
        # Zosilnenie pre silné výrazy
        strong_positive = ['surge', 'rally', 'breakout', 'soar', 'leap']
        strong_negative = ['plunge', 'crash', 'collapse', 'tumble', 'slump']
        
        for word in strong_positive:
            if word in text_lower:
                positive_score += 3
                
        for word in strong_negative:
            if word in text_lower:
                negative_score += 3
                
        total = positive_score + negative_score
        if total == 0:
            return 0
            
        # Vážený sentiment (-1 až 1)
        sentiment = (positive_score - negative_score) / total
        
        # Obmedzenie na rozsah -1 až 1
        return max(-1, min(1, sentiment))

if __name__ == "__main__":
    # Testovacia funkcia
    logging.basicConfig(level=logging.INFO)
    
    downloader = SentimentDownloader()
    
    # Test pre BTC
    print("\n" + "="*50)
    print("Testing BTC sentiment:")
    btc_sentiment = downloader.get_market_sentiment("BTCUSD_ecn", "crypto")
    print(btc_sentiment)
    
    # Test pre EURUSD
    print("\n" + "="*50)
    print("Testing EURUSD sentiment:")
    eurusd_sentiment = downloader.get_market_sentiment("EURUSD_ecn", "forex")
    print(eurusd_sentiment)
    
    # Test pre akciový index
    print("\n" + "="*50)
    print("Testing stock sentiment for DAX:")
    dax_sentiment = downloader.get_market_sentiment("DAX_ecn", "stock")
    print(dax_sentiment)
    
    # Test novinového sentimentu
    print("\n" + "="*50)
    print("Testing news sentiment for 'Bitcoin':")
    news_sentiment = downloader.get_news_sentiment("Bitcoin")
    print(news_sentiment)