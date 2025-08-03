import pandas as pd
from datetime import datetime, timedelta
from data_collection.mt5_data_loader import save_all_symbol_data  # Opravený import
from sentiment.sentiment_downloader import SentimentDownloader
from config import SYMBOLS, TIMEFRAME, ASSET_TYPES
import time
import os

def download_and_merge_data():
    # 1. Download historical price data
    save_all_symbol_data(TIMEFRAME)
    
    # 2. Initialize sentiment downloader
    sentiment_dl = SentimentDownloader()
    
    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        asset_type = ASSET_TYPES.get(symbol, 'stock')
        
        # Load price data
        file_name = f"{symbol}_{TIMEFRAME.replace('min', 'm')}.csv"
        price_df = pd.read_csv(f'data/{file_name}', 
                              parse_dates=['date'], 
                              index_col='date')
        
        # Generate daily sentiment
        sentiment_data = []
        unique_days = price_df.index.normalize().unique()
        
        for day in unique_days:
            next_day = day + timedelta(days=1)
            sentiment = sentiment_dl.get_market_sentiment(
                symbol, 
                asset_type, 
                from_time=day, 
                to_time=next_day - timedelta(seconds=1)
            )
            
            # Handle different sentiment response formats
            if sentiment and 'news_sentiment' in sentiment:
                news_sentiment = sentiment['news_sentiment']
                market_sentiment = sentiment.get('sentiment', 0)
            else:
                news_sentiment = 0
                market_sentiment = 0
            
            sentiment_data.append({
                'date': day,
                'market_sentiment': market_sentiment,
                'news_sentiment': news_sentiment
            })
            time.sleep(0.5)  # Rate limiting
        
        # Merge sentiment
        sentiment_df = pd.DataFrame(sentiment_data).set_index('date')
        merged_df = price_df.join(sentiment_df, how='left').ffill()
        
        # Save for training
        os.makedirs('training_data', exist_ok=True)
        merged_df.to_csv(f'training_data/{symbol}_with_sentiment.csv')
        print(f"Saved merged data for {symbol}")

if __name__ == "__main__":
    download_and_merge_data()