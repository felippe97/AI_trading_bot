# train_all.py
import os, sys
# přidejte kořenovou složku (kde je train_all.py) do PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from pathlib import Path
from training.advanced_models import AdvancedModelTrainer
from config import SYMBOLS, TIMEFRAME

def main():
    print(f"Starting training for {len(SYMBOLS)} symbols")
    
    for symbol in SYMBOLS:
        try:
            print(f"\n{'='*70}")
            print(f"Training model for {symbol} ({TIMEFRAME})")
            print(f"{'='*70}")
            
            # Build data path
            data_path = f"data/{symbol}_{TIMEFRAME}.csv"
            
            # Create trainer
            trainer = AdvancedModelTrainer(
                symbol=symbol,
                model_type='hybrid',
                data_path=data_path,
                timeframe=TIMEFRAME
            )
            
            # Run training
            trainer.train()
            print(f"Training completed for {symbol}")
            
        except Exception as e:
            print(f"Error during training {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()