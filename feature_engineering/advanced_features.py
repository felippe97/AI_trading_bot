# feature_engineering/advanced_features.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import requests
import os
from .macro_data_manager import MacroDataManager  # New unified data manager

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
        
        # Handle macro data based on mode
        self._handle_macro_data()
        
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
        # Existing seasonal feature code remains the same
        # ...
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
        if any(x in self.symbol for x in ['NSDQ', 'SP', 'TSLA', 'AAPL', 'NVDA']):
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
        df = self.add_seasonal_features(df)
        df = self.add_macro_features(df)
        df = self.add_geopolitical_risk(df)
        
        if 'BTC' in self.symbol:
            df = self.add_crypto_specific_features(df)
            
        return df