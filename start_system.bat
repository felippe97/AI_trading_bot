@echo off

start "MT5 Terminal" "C:\Program Files\MetaTrader 5\terminal64.exe"

start "Data Feed EURUSD" python data_collection/realtime_feed.py --symbol EURUSD_ecn --timeframe M5
timeout /t 5
start "Data Feed BTCUSD" python data_collection/realtime_feed.py --symbol BTCUSD_ecn --timeframe M5
timeout /t 5

start "Prediction Server" uvicorn predict_server:app --reload --port 8000
timeout /t 10

start "Trading Module" python execution/mt5_connector.py
timeout /t 5

start "Dashboard" python monitoring/performance_dashboard.py