# training/symbol_trainer.py
from advanced_models import AdvancedModelTrainer
from config import SYMBOLS, TIMEFRAME

def train_symbol(symbol, model_type):
    data_path = f"data/{symbol}_{TIMEFRAME}.csv"
    trainer = AdvancedModelTrainer(
        symbol=symbol,
        model_type=model_type,
        data_path=data_path,
        timeframe=TIMEFRAME
    )
    trainer.train()

if __name__ == "__main__":
    model_type = 'hybrid'  # or 'transfer'
    for symbol in SYMBOLS:
        print(f"\n{'='*50}")
        print(f"Training {symbol} with {model_type} model")
        print(f"{'='*50}")
        train_symbol(symbol, model_type)