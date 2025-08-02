# predict_server.py
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pathlib import Path
import sys
import joblib
import tensorflow as tf
import logging
from datetime import datetime, timedelta
import hashlib

# Pridajte koreňový priečinok do Python cesty
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from feature_engineering.intraday_features import compute_intraday_features
from training.online_calibration import RealTimeCalibrator
from config import TIMEFRAME, MODEL_PARAMS, SYMBOLS

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# Načítanie modelov a scalerov do pamäte pri štarte
models = {}
scalers = {}
history_data = {}
calibrators = {}
prediction_counters = {}

def load_models():
    """Načíta všetky modely a scalery do pamäte"""
    model_dir = root_dir / 'models'
    
    for symbol in SYMBOLS:
        model_path = model_dir / f"{symbol}_{TIMEFRAME}_model.keras"
        scaler_path = model_dir / f"{symbol}_{TIMEFRAME}_scaler.pkl"
        
        if model_path.exists() and scaler_path.exists():
            try:
                models[symbol] = tf.keras.models.load_model(model_path)
                scalers[symbol] = joblib.load(scaler_path)
                history_data[symbol] = pd.DataFrame()
                prediction_counters[symbol] = 0
                
                # Inicializácia kalibrátora
                calibrators[symbol] = RealTimeCalibrator(str(model_path))
                
                logger.info(f"Načítaný model a scaler pre {symbol}")
            except Exception as e:
                logger.error(f"Chyba pri načítavaní modelu pre {symbol}: {str(e)}")
        else:
            logger.warning(f"Chýbajúci model alebo scaler pre {symbol}")

@app.on_event("startup")
async def startup_event():
    load_models()

def should_calibrate(symbol, calibration_interval=50):
    """Určuje, či je čas na kalibráciu"""
    prediction_counters[symbol] += 1
    return prediction_counters[symbol] % calibration_interval == 0

@app.post("/predict/{symbol}")
async def predict(symbol: str, data: dict):
    try:
        # Kontrola existencie modelu
        if symbol not in models:
            raise HTTPException(status_code=404, detail=f"Model pre {symbol} nebol nájdený")
        
        # Konverzia dát do DataFrame
        df = pd.DataFrame([data])
        
        # Spracovanie časovej značky
        if 'time' in df:
            df['time'] = pd.to_datetime(df['time'])
        else:
            df['time'] = pd.Timestamp.now()
        
        # Nastavenie časového indexu
        df.set_index('time', inplace=True)
        
        # Aktualizácia histórie pre symbol
        if symbol not in history_data:
            history_data[symbol] = df.copy()
        else:
            # Kontrola duplicity pomocou časovej značky
            last_time = history_data[symbol].index[-1]
            if df.index[0] > last_time:
                history_data[symbol] = pd.concat([history_data[symbol], df])
        
        # Získanie dostatočnej histórie pre výpočet featureov
        min_history = max(MODEL_PARAMS['lookback_window'] * 2, 100)
        if len(history_data[symbol]) < min_history:
            raise HTTPException(
                status_code=422, 
                detail=f"Nedostatok historických dát. Potrebných aspoň {min_history} záznamov"
            )
        
        # Výpočet indikátorov na celej histórii
        df_with_features = compute_intraday_features(
            history_data[symbol].copy(), 
            timeframe=TIMEFRAME
        )
        
        # Odstránenie kategorických premenných
        features = df_with_features.drop(columns=['day_part'], errors='ignore')
        
        # Skalovanie dát
        scaler = scalers[symbol]
        scaled_data = scaler.transform(features)
        
        # Vytvorenie sekvencie pre model
        lookback = MODEL_PARAMS['lookback_window']
        sequence = scaled_data[-lookback:]
        sequence = sequence.reshape((1, sequence.shape[0], sequence.shape[1]))
        
        # Predikcia
        prediction = models[symbol].predict(sequence, verbose=0)
        class_id = np.argmax(prediction, axis=1)[0]
        
        # Online kalibrácia
        if should_calibrate(symbol):
            try:
                # Posledných 50 záznamov ako validačný set
                calibration_data = history_data[symbol].iloc[-50:]
                
                # Kontrola, či máme dosť dát pre kalibráciu
                if len(calibration_data) > lookback:
                    df_cal_features = compute_intraday_features(
                        calibration_data.copy(), 
                        timeframe=TIMEFRAME
                    ).drop(columns=['day_part'], errors='ignore')
                    
                    X_cal = scaler.transform(df_cal_features)
                    
                    # Vytvorenie sekvencií pre kalibráciu
                    X_seq_cal, y_seq_cal = [], []
                    for i in range(lookback, len(X_cal)):
                        X_seq_cal.append(X_cal[i-lookback:i])
                        
                        # Výpočet výnosu medzi barami
                        future_ret = calibration_data['close'].iloc[i] / calibration_data['close'].iloc[i-1] - 1
                        label = 1  # HOLD
                        if future_ret > MODEL_PARAMS['threshold']:
                            label = 2  # BUY
                        elif future_ret < -MODEL_PARAMS['threshold']:
                            label = 0  # SELL
                        y_seq_cal.append(label)
                    
                    X_seq_cal = np.array(X_seq_cal)
                    y_seq_cal = np.array(y_seq_cal)
                    
                    # Kalibrácia
                    loss = calibrators[symbol].calibrate(X_seq_cal, y_seq_cal)
                    
                    # Kontrola degradácie
                    calibrators[symbol].revert_if_degraded(
                        validation_data=(X_seq_cal, y_seq_cal)
                    )
                    
                    logger.info(f"Online kalibrácia pre {symbol}: str={loss:.4f}")
                    
                    # Aktualizácia modelu v pamäti
                    models[symbol] = calibrators[symbol].model
            except Exception as e:
                logger.error(f"Chyba pri kalibrácii: {str(e)}")
        
        return {
            "symbol": symbol,
            "prediction": int(class_id),
            "confidence": float(prediction[0][class_id]),
            "class_names": ["SELL", "HOLD", "BUY"]
        }
    except Exception as e:
        logger.exception("Chyba pri predikcii")
        raise HTTPException(status_code=500, detail=str(e))

# Tento blok musí byť mimo funkcie!
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)