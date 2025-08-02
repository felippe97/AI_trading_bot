# feature_engineering/macro_data_downloader.py
import pandas as pd
import requests
import os
from datetime import datetime

class MacroDataDownloader:
    def __init__(self, data_dir="data/macro"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_btc_dominance(self, start_date="2010-01-01"):
        """Stiahne historické dáta o dominancii Bitcoinu"""
        url = f"https://api.alternative.me/v2/dominance/?since={start_date}&limit=0"
        response = requests.get(url).json()
        data = response['data']
        df = pd.DataFrame({
            'Date': [datetime.utcfromtimestamp(d['timestamp']) for d in data],
            'dominance': [d['dominance_percentage'] for d in data]
        })
        df.to_csv(f"{self.data_dir}/btc_dominance.csv", index=False)
        return df

    def download_vix(self, start_date="1990-01-01"):
        """Stiahne historické dáta VIX indexu"""
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS&cosd={start_date}"
        df = pd.read_csv(url, parse_dates=['DATE'])
        df.columns = ['Date', 'VIX']
        df.to_csv(f"{self.data_dir}/VIX.csv", index=False)
        return df

    def download_gold_reserves(self):
        """Stiahne dáta o zlatých rezervách"""
        # Príklad reálneho API (vyžaduje API kľúč)
        # url = "https://www.goldapi.io/api/reserves"
        # headers = {"x-access-token": "YOUR_API_KEY"}
        # response = requests.get(url, headers=headers).json()
        
        # Dummy dáta pre demonštráciu:
        df = pd.DataFrame({
            'Date': pd.date_range(start='2000-01-01', end=datetime.today(), freq='M'),
            'reserves': np.random.uniform(30000, 35000, 300)
        })
        df.to_csv(f"{self.data_dir}/gold_reserves.csv", index=False)
        return df

    def download_all_historical(self):
        """Stiahne všetky potrebné historické makro dáta"""
        print("Sťahujem historické makro dáta...")
        self.download_btc_dominance()
        self.download_vix()
        self.download_gold_reserves()
        print("Makro dáta úspešne stiahnuté!")

# Použitie pred tréningom:
if __name__ == "__main__":
    downloader = MacroDataDownloader()
    downloader.download_all_historical()