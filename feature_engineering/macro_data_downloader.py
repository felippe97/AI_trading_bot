# feature_engineering/macro_data_manager.py
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
import logging

# KRIITICKÁ OPRAVA: Pridaná inicializácia loggera
logger = logging.getLogger(__name__)

class MacroDataManager:
    def __init__(self, data_dir="data/macro"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.btc_dominance_path = os.path.join(data_dir, "btc_dominance.csv")
        self.vix_path = os.path.join(data_dir, "VIX.csv")
        self.oil_inventory_path = os.path.join(data_dir, "oil_inventory.csv")
        self.gold_reserves_path = os.path.join(data_dir, "gold_reserves.csv")
        
    def download_btc_dominance(self):
        """Stiahne historické dáta o dominancii Bitcoinu"""
        try:
            # OPRAVENÉ URL - bez dátového parametra
            url = "https://api.alternative.me/v2/dominance/?limit=1000"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json().get('data', [])
            
            if not data:
                logger.warning("No data in BTC dominance response")
                return None
                
            df = pd.DataFrame({
                'Date': [datetime.utcfromtimestamp(d['timestamp']) for d in data],
                'dominance': [d['dominance_percentage'] for d in data]
            })
            
            df.to_csv(self.btc_dominance_path, index=False)
            logger.info("BTC dominance data downloaded successfully")
            return df
        except Exception as e:
            logger.error(f"Error downloading BTC dominance: {str(e)}")
            return None

    def download_vix(self):
        """Stiahne historické dáta VIX indexu"""
        try:
            # OPRAVENÝ FORMÁT - bez parsovania konkrétneho stĺpca
            url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"
            df = pd.read_csv(url)
            
            # Automatická detekcia stĺpcov
            if len(df.columns) < 2:
                logger.error("VIX data has less than 2 columns")
                return None
                
            # Premenovanie stĺpcov
            df.columns = ['Date', 'VIX'] 
            df['Date'] = pd.to_datetime(df['Date'])
            df.to_csv(self.vix_path, index=False)
            logger.info("VIX data downloaded successfully")
            return df
        except Exception as e:
            logger.error(f"Error downloading VIX: {str(e)}")
            return None

    def download_oil_inventory(self):
        """Stiahne dáta o zásobách ropy"""
        try:
            from config import EIA_API_KEY
            
            if not EIA_API_KEY:
                logger.warning("EIA_API_KEY not found in config")
                return None
                
            url = f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/?api_key={EIA_API_KEY}&frequency=weekly&data[0]=value&facets[series][]=WCRSTUS1&sort[0][column]=period&sort[0][direction]=desc"
            
            response = requests.get(url)
            response.raise_for_status()
            data = response.json().get('response', {}).get('data', [])
            
            if not data:
                logger.warning("No data in oil inventory response")
                return None
                
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['period'])
            df = df.sort_values('Date')
            df = df[['Date', 'value']].rename(columns={'value': 'oil_inventory'})
            df.to_csv(self.oil_inventory_path, index=False)
            logger.info("Oil inventory data downloaded successfully")
            return df
        except Exception as e:
            logger.error(f"Error downloading oil inventory: {str(e)}")
            return None

    def download_gold_reserves(self):
        """Stiahne dáta o zlatých rezervách"""
        try:
            # Dummy dáta
            df = pd.DataFrame({
                'Date': pd.date_range(start='2000-01-01', end=datetime.today(), freq='M'),
                'reserves': np.random.uniform(30000, 35000, 300)
            })
            df.to_csv(self.gold_reserves_path, index=False)
            logger.info("Gold reserves data generated")
            return df
        except Exception as e:
            logger.error(f"Error generating gold reserves: {str(e)}")
            return None

    def download_all_historical(self):
        """Stiahne všetky potrebné historické makro dáta"""
        logger.info("Downloading historical macro data...")
        
        # OPRAVA: Odstránený problémový parameter z URL
        self.download_btc_dominance()
        self.download_vix()
        self.download_oil_inventory()
        self.download_gold_reserves()
        
        logger.info("Macro data download completed!")

    def get_btc_dominance(self):
        """Načíta dáta o BTC dominancii"""
        if os.path.exists(self.btc_dominance_path):
            df = pd.read_csv(self.btc_dominance_path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            return df
        return None

    def get_vix(self):
        """Načíta dáta VIX indexu"""
        if os.path.exists(self.vix_path):
            df = pd.read_csv(self.vix_path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            return df
        return None

    def get_oil_inventory(self):
        """Načíta dáta o zásobách ropy"""
        if os.path.exists(self.oil_inventory_path):
            df = pd.read_csv(self.oil_inventory_path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            return df
        return None

    def get_gold_reserves(self):
        """Načíta dáta o zlatých rezervách"""
        if os.path.exists(self.gold_reserves_path):
            df = pd.read_csv(self.gold_reserves_path, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            return df
        return None

    def update_all(self):
        """Aktualizuje všetky makro dáta"""
        logger.info("Updating macro data...")
        self.download_all_historical()