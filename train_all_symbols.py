# train_all.py (v koreňovom adresári)
import sys
import os
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Absolútna cesta k priečinku training
training_dir = Path(__file__).parent / "training"
sys.path.insert(0, str(training_dir))

# Importujte AdvancedModelTrainer priamo
from advanced_models import AdvancedModelTrainer

# Import konfigurácie
sys.path.insert(0, str(Path(__file__).parent))
from config import SYMBOLS, TIMEFRAME

def main():
    print(f"Starting training for {len(SYMBOLS)} symbols")
    
    for symbol in SYMBOLS:
        try:
            print(f"\n{'='*70}")
            print(f"Training model for {symbol} ({TIMEFRAME})")
            print(f"{'='*70}")
            
            # Zostavte cestu k dátam
            data_path = Path("data") / f"{symbol}_{TIMEFRAME}.csv"
            
            # Skontrolujte existenciu súboru
            if not data_path.exists():
                print(f"⚠️ Data file not found: {data_path}")
                continue
                
            # Vytvorte inštanciu trénera
            trainer = AdvancedModelTrainer(
                symbol=symbol,
                model_type='hybrid',
                data_path=str(data_path),
                timeframe=TIMEFRAME
            )
            
            # Spustite tréning
            trainer.train()
            print(f"✅ Training completed for {symbol}")
            
        except Exception as e:
            print(f"🔥 Error during training {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Pokračujte v tréningu ďalších symbolov
            continue

if __name__ == "__main__":
    main()