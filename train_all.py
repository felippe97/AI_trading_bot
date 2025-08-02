# train_all.py (aktualizované o automatické sťahovanie dát)
import sys
import os
import logging
from datetime import datetime
from data_downloader import DataDownloader  # Import the new downloader

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)
# Get the root directory (two levels up from the script)
project_root = os.path.dirname(os.path.dirname(current_script_path))
# Add root to Python path
sys.path.insert(0, project_root)

from training.advanced_models import AdvancedModelTrainer
from config import SYMBOLS, TIMEFRAME, HISTORICAL_BARS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_data_exists():
    """Ensure data files exist, download if missing"""
    downloader = DataDownloader()
    return downloader.save_all_data()

def main():
    # Ensure data exists before training
    if not ensure_data_exists():
        logger.error("Data preparation failed. Training aborted.")
        return
        
    logger.info(f"Starting training for {len(SYMBOLS)} symbols")
    
    for symbol in SYMBOLS:
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"Training model for {symbol} ({TIMEFRAME})")
            logger.info(f"{'='*70}")
            
            # Build data path
            data_path = f"data/{symbol}_{TIMEFRAME}.csv"
            full_data_path = os.path.join(project_root, data_path)
            
            # Skip if data file doesn't exist
            if not os.path.exists(full_data_path):
                logger.warning(f"Data file not found: {full_data_path}. Skipping {symbol}.")
                continue
                
            # Create trainer
            trainer = AdvancedModelTrainer(
                symbol=symbol,
                model_type='hybrid',
                data_path=full_data_path,
                timeframe=TIMEFRAME
            )
            
            # Run training
            trainer.train()
            logger.info(f"Training completed for {symbol}")
            
        except Exception as e:
            logger.error(f"Error during training {symbol}: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()