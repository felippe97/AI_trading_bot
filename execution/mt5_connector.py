# execution/mt5_connector.py

import zmq
import time
import json
import numpy as np
from execution.risk_manager import calculate_position_size
import MetaTrader5 as mt5
from execution.risk_manager import calculate_position_size
def initialize_mt5():
    # ZADAJTE VAŠE PRIHLASOVACIE ÚDAJE
    account = 20045680       # Číslo vášho účtu
    password = "Eltequito1@"  # Heslo k účtu
    server = "PurpleTrading-Demo" # Názov servera brokera (napr. "RoboForex-ECN")
    
    if not mt5.initialize():
        print("Initialize failed, error code =", mt5.last_error())
        return False
    
    # Pripojenie k účtu
    authorized = mt5.login(account, password=password, server=server)
    
    if not authorized:
        print("Login failed, error code =", mt5.last_error())
        return False
    
    print("Úspešne pripojené k účtu:", account)
    return True

def execute_trade(signal, symbol, balance, atr):
    if not initialize_mt5():
        return False
    
    # Výpočet veľkosti pozície
    symbol_info = mt5.symbol_info(symbol)
    point = symbol_info.point
    pip_value = point * 10  # Pre forex páry
    
    sl_points = atr * 0.67  # 2/3 ATR
    tp_points = sl_points * 1.5  # Pomer 1.5:1
    
    volume = calculate_position_size(balance, RISK_PER_TRADE, sl_points, pip_value)
    
    # Príprava objednávky
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY if signal == 2 else mt5.ORDER_TYPE_SELL,
        "price": mt5.symbol_info_tick(symbol).ask if signal == 2 else mt5.symbol_info_tick(symbol).bid,
        "sl": sl_points,
        "tp": tp_points,
        "deviation": 10,
        "magic": 202308,
        "comment": "AI Intraday System",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK
    }
    
    # Odoslanie objednávky
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Trade execution failed for {symbol}: {result.comment}")
        return False
    return True

def start_trading_loop():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    socket.connect("tcp://localhost:5555")  # Pripojenie k dátovému feedu
    
    account_balance = mt5.account_info().balance
    
    while True:
        try:
            data = socket.recv_pyobj()
            symbol = data['symbol']
            
            # TU BY BOLO VOLANIE MODELU PRE PREDIKCIU
            # signal = model.predict(data)
            
            # execute_trade(signal, symbol, account_balance, data['atr'])
            
            # Monitorovanie drawdownu
            current_equity = mt5.account_info().equity
            if ((account_balance - current_equity) / account_balance) * 100 >= MAX_DAILY_DRAWDOWN:
                print("Daily drawdown limit reached! Stopping trading.")
                break
                
        except Exception as e:
            print(f"Error in trading loop: {e}")
        time.sleep(0.1)