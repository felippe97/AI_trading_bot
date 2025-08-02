# run_training.py
import os
from training.advanced_models import AdvancedModelTrainer
from config import SYMBOLS, TIMEFRAME
# Pridajte koreňový adresár do Python path
sys.path.append(str(Path(__file__).parent))

from training.advanced_models import AdvancedModelTrainer
from config import SYMBOLS, TIMEFRAME
def main():
    for symbol in SYMBOLS:
        try:
            print(f"\n{'='*50}")
            print(f"Starting training for {symbol} ({TIMEFRAME})")
            print(f"{'='*50}")
            
            # Cesta k dátam
            data_path = f"data/{symbol}_{TIMEFRAME}.csv"
            
            # Skontrolujte existenciu súboru
            if not os.path.exists(data_path):
                print(f"Data file not found: {data_path}")
                continue
                
            # Vytvorenie inštancie trénera
            trainer = AdvancedModelTrainer(
                symbol=symbol,
                model_type='hybrid',  # Alebo 'transfer'
                data_path=data_path,
                timeframe=TIMEFRAME
            )
            
            # Spustenie tréningu
            trainer.train()
            
            print(f"Training completed for {symbol}")
            
        except Exception as e:
            print(f"Error during training {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()