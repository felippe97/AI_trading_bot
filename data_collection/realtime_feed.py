# data_collection/realtime_feed.py
import sys
import os
from pathlib import Path
import zmq
import time
import MetaTrader5 as mt5

# Pridajte koreňový priečinok do Python cesty
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from config import SYMBOLS

def start_realtime_feed(symbol, timeframe=mt5.TIMEFRAME_M5, port=5555):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return
    
    print(f"Starting real-time feed for {symbol} on timeframe {timeframe}")
    
    # Počiatočný stav
    last_bar = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)[0]
    last_time = last_bar['time']
    
    while True:
        new_bars = mt5.copy_rates_from(symbol, timeframe, last_time, 1)
        if new_bars is not None and len(new_bars) > 0:
            new_bar = new_bars[0]
            if new_bar['time'] != last_time:
                # Poslať nový bar
                socket.send_pyobj({
                    'symbol': symbol,
                    'time': new_bar['time'],
                    'open': new_bar['open'],
                    'high': new_bar['high'],
                    'low': new_bar['low'],
                    'close': new_bar['close'],
                    'volume': new_bar['tick_volume']
                })
                last_time = new_bar['time']
        time.sleep(0.1)  # Pauza 100ms

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--timeframe', default='M5')
    args = parser.parse_args()
    
    # Mapovanie časového rámca
    timeframe_map = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1
    }
    
    start_realtime_feed(
        symbol=args.symbol,
        timeframe=timeframe_map.get(args.timeframe, mt5.TIMEFRAME_M5)
    )