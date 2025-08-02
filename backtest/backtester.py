# backtest/backtester.py
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Pridajte koreňový priečinok do Python cesty
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Riešenie problému s importom feature_engineering
try:
    from feature_engineering.intraday_features import compute_intraday_features
except ImportError:
    # Alternatívna cesta pre import
    sys.path.append(str(root_dir / "feature_engineering"))
    from intraday_features import compute_intraday_features

from config import TIMEFRAME, MODEL_PARAMS, SYMBOLS

class Backtester:
    def __init__(self, symbol, data_path, initial_balance=10000, fee=0.0002):
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.fee = fee  # poplatok za obchod
        self.data = self._load_data(data_path)
        self.model = None
        self.scaler = None
        self.lookback = MODEL_PARAMS['lookback_window']
        self.future_bars = MODEL_PARAMS['future_bars']
        
    def _load_data(self, path):
        df = pd.read_csv(path, parse_dates=['time'], index_col='time')
        
        # Overenie existencie stĺpca volume
        if 'volume' not in df.columns and 'tick_volume' in df.columns:
            df = df.rename(columns={'tick_volume': 'volume'})
        
        # Resampling na cieľový timeframe
        timeframe_str = TIMEFRAME.replace('T', 'min')
        df = df.resample(timeframe_str).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return compute_intraday_features(df, TIMEFRAME)
    
    def load_model(self):
        model_dir = root_dir / 'models'
        model_path = model_dir / f"{self.symbol}_{TIMEFRAME}_model.keras"
        scaler_path = model_dir / f"{self.symbol}_{TIMEFRAME}_scaler.pkl"
        
        if model_path.exists() and scaler_path.exists():
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Načítaný model a scaler pre {self.symbol}")
        else:
            raise FileNotFoundError(f"Model alebo scaler pre {self.symbol} nebol nájdený")
    
    def create_labels(self, threshold=0.001):
        future_ret = self.data['close'].pct_change(self.future_bars).shift(-self.future_bars)
        labels = pd.cut(future_ret, 
                       bins=[-np.inf, -threshold, threshold, np.inf],
                       labels=[0, 1, 2])  # 0=SELL, 1=HOLD, 2=BUY
        return labels.dropna()
    
    def prepare_dataset(self):
        # Odstránime kategorickú premennú
        features = self.data.drop(columns=['day_part'], errors='ignore')
        labels = self.create_labels()
        
        # Spojenie features a labels
        merged = pd.concat([features, labels], axis=1).dropna()
        X = merged.iloc[:, :-1]
        y = merged.iloc[:, -1]
        
        # Skalovanie
        X_scaled = self.scaler.transform(X)
        
        # Vytvorenie sekvencií
        X_seq, y_seq = [], []
        for i in range(self.lookback, len(X_scaled) - self.future_bars):
            X_seq.append(X_scaled[i-self.lookback:i])
            y_seq.append(y.iloc[i])
            
        return np.array(X_seq), np.array(y_seq), merged.index[self.lookback:len(X_scaled)-self.future_bars]
    
    def run_backtest(self):
        if not self.model or not self.scaler:
            self.load_model()
        
        X, y, timestamps = self.prepare_dataset()
        predictions = self.model.predict(X, verbose=0)
        class_predictions = np.argmax(predictions, axis=1)
        
        # Výpočet metrík
        print(f"\nVýsledky pre {self.symbol}:")
        print(classification_report(y, class_predictions, target_names=['SELL', 'HOLD', 'BUY']))
        print("Matica zámen:")
        print(confusion_matrix(y, class_predictions))
        
        # Simulácia obchodovania
        cash = self.initial_balance
        position = 0  # 0: flat, 1: long, -1: short
        entry_price = 0
        equity = [cash]
        portfolio_values = []
        trades = []
        
        for i in range(len(X)):
            current_close = self.data.loc[timestamps[i]]['close']
            pred = class_predictions[i]
            
            # Obchodná logika
            if pred == 2:  # BUY signál
                if position == 0:
                    # Vstup do LONG
                    position = 1
                    entry_price = current_close
                    cash -= cash * self.fee  # poplatok
                    trades.append(('BUY', timestamps[i], current_close))
                elif position == -1:
                    # Uzavretie SHORT a vstup do LONG
                    # Zisk z short
                    pnl = (entry_price - current_close) / entry_price * cash
                    cash += pnl
                    cash -= cash * self.fee  # poplatok
                    
                    # Vstup do long
                    entry_price = current_close
                    cash -= cash * self.fee  # poplatok
                    position = 1
                    trades.append(('BUY', timestamps[i], current_close))
            
            elif pred == 0:  # SELL signál
                if position == 0:
                    # Vstup do SHORT
                    position = -1
                    entry_price = current_close
                    cash -= cash * self.fee  # poplatok
                    trades.append(('SELL', timestamps[i], current_close))
                elif position == 1:
                    # Uzavretie LONG a vstup do SHORT
                    # Zisk z long
                    pnl = (current_close - entry_price) / entry_price * cash
                    cash += pnl
                    cash -= cash * self.fee  # poplatok
                    
                    # Vstup do short
                    entry_price = current_close
                    cash -= cash * self.fee  # poplatok
                    position = -1
                    trades.append(('SELL', timestamps[i], current_close))
            
            # Výpočet hodnoty portfólia
            if position == 1:  # LONG
                pnl = (current_close - entry_price) / entry_price * cash
                portfolio_value = cash + pnl
            elif position == -1:  # SHORT
                pnl = (entry_price - current_close) / entry_price * cash
                portfolio_value = cash + pnl
            else:  # FLAT
                portfolio_value = cash
            
            equity.append(portfolio_value)
            portfolio_values.append(portfolio_value)
        
        # Uzavretie pozície na konci
        last_close = self.data['close'].iloc[-1]
        if position == 1:
            pnl = (last_close - entry_price) / entry_price * cash
            cash += pnl
            cash -= cash * self.fee
        elif position == -1:
            pnl = (entry_price - last_close) / entry_price * cash
            cash += pnl
            cash -= cash * self.fee
        equity.append(cash)
        
        # Výpočet výkonnosti
        portfolio_series = pd.Series(portfolio_values)
        returns = portfolio_series.pct_change().fillna(0)
        total_return = (equity[-1] / self.initial_balance - 1) * 100
        
        # Sharpe Ratio s kontrolou delenia nulou
        if returns.std() > 0:
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
            
        max_drawdown = (portfolio_series / portfolio_series.cummax() - 1).min() * 100
        
        print(f"\nVýsledky obchodovania pre {self.symbol}:")
        print(f"Počiatočný kapitál: ${self.initial_balance:.2f}")
        print(f"Konečný zostatok: ${equity[-1]:.2f}")
        print(f"Celkový výnos: {total_return:.2f}%")
        print(f"Počet obchodov: {len(trades)}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        
        # Vizualizácia
        plt.figure(figsize=(12, 8))
        
        # Graf výkonnosti
        plt.subplot(2, 1, 1)
        plt.plot(equity)
        plt.title(f"Výkonnosť portfólia: {self.symbol}")
        plt.ylabel("Hodnota portfólia ($)")
        plt.grid(True)
        
        # Graf predikcií vs realita
        plt.subplot(2, 1, 2)
        plt.plot(y, 'b-', label='Skutočné hodnoty')
        plt.plot(class_predictions, 'r--', alpha=0.7, label='Predikcie')
        plt.title("Porovnanie predikcií a skutočných hodnôt")
        plt.xlabel("Časový krok")
        plt.ylabel("Signál")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        results_dir = root_dir / "backtest" / "results"
        results_dir.mkdir(exist_ok=True, parents=True)
        plt.savefig(results_dir / f"{self.symbol}_backtest.png")
        plt.close()
        
        # Uloženie výsledkov
        results = {
            'symbol': self.symbol,
            'initial_balance': self.initial_balance,
            'final_balance': equity[-1],
            'total_return': total_return,
            'num_trades': len(trades),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades
        }
        
        return results

def run_all_backtests():
    results = []
    for symbol in SYMBOLS:
        try:
            print(f"\n{'='*50}")
            print(f"Spúšťam backtest pre {symbol}")
            print(f"{'='*50}")
            
            data_path = root_dir / 'data' / f"{symbol}_M1.csv"
            if not data_path.exists():
                print(f"Chýbajúce dáta pre {symbol}! Súbor {data_path} neexistuje.")
                continue
            
            backtester = Backtester(symbol, str(data_path))
            result = backtester.run_backtest()
            results.append(result)
            
            # Uloženie výsledkov pre symbol
            result_dir = root_dir / 'backtest' / 'results'
            result_dir.mkdir(exist_ok=True, parents=True)
            result_path = result_dir / f"{symbol}_backtest_results.csv"
            
            # Vytvorenie DataFrame z obchodov
            trade_data = []
            for trade in result['trades']:
                trade_data.append({
                    'timestamp': trade[1],
                    'action': trade[0],
                    'price': trade[2]
                })
            
            result_df = pd.DataFrame(trade_data)
            result_df.to_csv(result_path, index=False)
            
            print(f"Výsledky pre {symbol} uložené: {result_path}")
            
        except Exception as e:
            print(f"Chyba pri backteste pre {symbol}: {str(e)}")
    
    # Celkové štatistiky
    if results:
        print("\nCelkové výsledky:")
        summary_data = []
        for res in results:
            summary_data.append({
                'symbol': res['symbol'],
                'initial_balance': res['initial_balance'],
                'final_balance': res['final_balance'],
                'total_return': res['total_return'],
                'num_trades': res['num_trades'],
                'sharpe_ratio': res['sharpe_ratio'],
                'max_drawdown': res['max_drawdown']
            })
        
        summary = pd.DataFrame(summary_data)
        print(summary)
        
        # Uloženie súhrnu
        summary_dir = root_dir / 'backtest'
        summary_dir.mkdir(exist_ok=True, parents=True)
        summary_path = summary_dir / 'overall_summary.csv'
        summary.to_csv(summary_path, index=False)
        print(f"\nSúhrnné výsledky uložené: {summary_path}")
    else:
        print("Žiadne výsledky na zobrazenie")

if __name__ == "__main__":
    run_all_backtests()