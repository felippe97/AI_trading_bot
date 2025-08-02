# feature_engineering/macro_data_manager.py
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import time
import logging

class MacroDataManager:
    def __init__(self, data_dir="data/macro"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.api_key = os.getenv("MACRO_API_KEY", "YOUR_DEFAULT_API_KEY")
        self.logger = logging.getLogger(__name__)
        
    def download_all_historical(self):
        """Download all historical macro data"""
        self.logger.info("Downloading historical macro data...")
        self.download_btc_dominance()
        self.download_vix()
        self.download_oil_inventory()
        self.download_gold_reserves()
        self.download_geopolitical_risk()
        self.logger.info("Historical macro data download complete")
    
    def update_all(self):
        """Update all macro data to latest values"""
        self.logger.info("Updating macro data...")
        self.update_btc_dominance()
        self.update_vix()
        self.update_oil_inventory()
        self.update_gold_reserves()
        self.update_geopolitical_risk()
        self.logger.info("Macro data update complete")
    
    # BTC Dominance
    def download_btc_dominance(self, start_date="2010-01-01"):
        """Download BTC dominance historical data"""
        url = f"https://api.alternative.me/v2/dominance/?since={start_date}&limit=0"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()['data']
            df = pd.DataFrame({
                'Date': [datetime.utcfromtimestamp(d['timestamp']) for d in data],
                'dominance': [d['dominance_percentage'] for d in data]
            })
            df.to_csv(f"{self.data_dir}/btc_dominance.csv", index=False)
        except Exception as e:
            self.logger.error(f"Error downloading BTC dominance: {str(e)}")
    
    def update_btc_dominance(self):
        """Update BTC dominance data"""
        try:
            # Load existing data
            file_path = f"{self.data_dir}/btc_dominance.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, parse_dates=['Date'])
                last_date = df['Date'].max()
            else:
                df = pd.DataFrame(columns=['Date', 'dominance'])
                last_date = datetime(2010, 1, 1)
            
            # Get new data if needed
            if datetime.utcnow() - last_date > timedelta(days=1):
                url = "https://api.alternative.me/v2/global/"
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                new_data = response.json()['data']
                
                new_row = {
                    'Date': datetime.utcnow(),
                    'dominance': new_data['bitcoin_dominance_percentage']
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.to_csv(file_path, index=False)
        except Exception as e:
            self.logger.error(f"Error updating BTC dominance: {str(e)}")
    
    def get_btc_dominance(self):
        """Get BTC dominance data"""
        df = pd.read_csv(f"{self.data_dir}/btc_dominance.csv", parse_dates=['Date'])
        return df.set_index('Date')['dominance'].rename('btc_dominance')
    
    # VIX Index
    def download_vix(self, start_date="1990-01-01"):
        """Download historical VIX data"""
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS&cosd={start_date}"
            df = pd.read_csv(url, parse_dates=['DATE'])
            df.columns = ['Date', 'VIX']
            df.to_csv(f"{self.data_dir}/vix.csv", index=False)
        except Exception as e:
            self.logger.error(f"Error downloading VIX: {str(e)}")
    
    def update_vix(self):
        """Update VIX data"""
        try:
            # VIX data from FRED updates daily automatically
            # Just redownload the entire dataset for simplicity
            self.download_vix()
        except Exception as e:
            self.logger.error(f"Error updating VIX: {str(e)}")
    
    def get_vix(self):
        """Get VIX data"""
        df = pd.read_csv(f"{self.data_dir}/vix.csv", parse_dates=['Date'])
        return df.set_index('Date')['VIX']
    
    # Oil Inventory - using EIA API
    def download_oil_inventory(self, start_date="2000-01-01"):
        """Download historical oil inventory data"""
        try:
            url = f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/?frequency=daily&data[0]=value&start={start_date}&end={datetime.now().strftime('%Y-%m-%d')}&api_key={self.api_key}"
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()['response']['data']
            df = pd.DataFrame(data)[['period', 'value']]
            df.columns = ['Date', 'oil_inventory']
            df['Date'] = pd.to_datetime(df['Date'])
            df.to_csv(f"{self.data_dir}/oil_inventory.csv", index=False)
        except Exception as e:
            self.logger.error(f"Error downloading oil inventory: {str(e)}")
    
    def update_oil_inventory(self):
        """Update oil inventory data"""
        try:
            file_path = f"{self.data_dir}/oil_inventory.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, parse_dates=['Date'])
                last_date = df['Date'].max()
            else:
                df = pd.DataFrame(columns=['Date', 'oil_inventory'])
                last_date = datetime(2000, 1, 1)
            
            if datetime.now() - last_date > timedelta(days=7):
                self.download_oil_inventory(start_date=(last_date + timedelta(days=1)).strftime('%Y-%m-%d'))
        except Exception as e:
            self.logger.error(f"Error updating oil inventory: {str(e)}")
    
    def get_oil_inventory(self):
        """Get oil inventory data"""
        df = pd.read_csv(f"{self.data_dir}/oil_inventory.csv", parse_dates=['Date'])
        return df.set_index('Date')['oil_inventory']
    
    # Gold Reserves - using World Gold Council data
    def download_gold_reserves(self, start_date="2000-01-01"):
        """Download gold reserves data"""
        try:
            # This would use actual API in production
            # For demo, generate synthetic data
            dates = pd.date_range(start=start_date, end=datetime.now(), freq='M')
            values = np.random.uniform(30000, 35000, len(dates))
            df = pd.DataFrame({'Date': dates, 'gold_reserves': values})
            df.to_csv(f"{self.data_dir}/gold_reserves.csv", index=False)
        except Exception as e:
            self.logger.error(f"Error downloading gold reserves: {str(e)}")
    
    def update_gold_reserves(self):
        """Update gold reserves data"""
        try:
            file_path = f"{self.data_dir}/gold_reserves.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, parse_dates=['Date'])
                last_date = df['Date'].max()
            else:
                self.download_gold_reserves()
                return
            
            if datetime.now() - last_date > timedelta(days=30):
                self.download_gold_reserves(start_date=last_date.strftime('%Y-%m-%d'))
        except Exception as e:
            self.logger.error(f"Error updating gold reserves: {str(e)}")
    
    def get_gold_reserves(self):
        """Get gold reserves data"""
        df = pd.read_csv(f"{self.data_dir}/gold_reserves.csv", parse_dates=['Date'])
        return df.set_index('Date')['gold_reserves']
    
    # Geopolitical Risk Index
    def download_geopolitical_risk(self):
        """Download geopolitical risk data"""
        try:
            # In production, this would use actual API
            # For demo, generate synthetic data
            dates = pd.date_range(start='2000-01-01', end=datetime.now(), freq='D')
            values = np.random.randint(0, 100, len(dates))
            df = pd.DataFrame({'Date': dates, 'gpr_index': values})
            df.to_csv(f"{self.data_dir}/geopolitical_risk.csv", index=False)
        except Exception as e:
            self.logger.error(f"Error downloading geopolitical risk: {str(e)}")
    
    def update_geopolitical_risk(self):
        """Update geopolitical risk data"""
        try:
            self.download_geopolitical_risk()  # Daily full refresh
        except Exception as e:
            self.logger.error(f"Error updating geopolitical risk: {str(e)}")
    
    def get_geopolitical_risk(self):
        """Get geopolitical risk data"""
        df = pd.read_csv(f"{self.data_dir}/geopolitical_risk.csv", parse_dates=['Date'])
        return df.set_index('Date')['gpr_index']