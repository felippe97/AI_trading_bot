# config.py
SYMBOLS = [
    'BTCUSD_ecn', 'DAX_ecn', 'EURUSD_ecn', 
    'NSDQ_ecn', 'SP_ecn', 'USOIL.fut', 'XAUUSD_ecn'
]

TIMEFRAME = '5min'  # Zmenené z '5T' na '5min' kvôli kompatibilite s Pandas
HISTORICAL_BARS = 50000
MAX_DAILY_DRAWDOWN = 5.0  # %
RISK_PER_TRADE = 1.0  # % z účtu
TP_SL_RATIO = 1.5  # Pomer TP/SL

# Základné parametre modelu
MODEL_PARAMS = {
    'epochs': 100,
    'batch_size': 128,
    'lookback_window': 60,  # Počet sviečok spätne
    'future_bars': 3,  # Predpoveď o 3 sviečky dopredu
    'threshold': 0.0015,  # Hranica pre obchodný signál
    'lstm_units': [96, 64],  # Štandardné veľkosti LSTM vrstiev
    'model_type': 'hybrid'  # 'hybrid' alebo 'transfer'
}

# Symbol-špecifické hyperparametre
SYMBOL_HYPERPARAMS = {
    'BTCUSD_ecn': {
        'lookback_window': 90,   # Väčšie okno pre volatilné aktívum
        'threshold': 0.0020,     # Vyšší threshold kvôli vyššej volatilite
        'lstm_units': [128, 96], # Hlbšia sieť
        'risk_multiplier': 1.5   # Vyššie riziko pre volatilné aktívum
    },
    'EURUSD_ecn': {
        'lookback_window': 45,
        'threshold': 0.0010,     # Nižší threshold pre menšie pohyby
        'lstm_units': [80, 48],
        'risk_multiplier': 0.8   # Nižšie riziko pre stabilnejší trh
    },
    'USOIL.fut': {
    'lookback_window': 120,  # Zvýšte na 100-120 pre lepšiu históriu
    'threshold': 0.0025,     # Zvýšte pre lepšiu identifikáciu trendov
    'lstm_units': [150, 100, 50],  # Pridajte tretiu vrstvu
    'dropout_rate': 0.5,     # Zvýšte na potlačenie overfittingu
    'epochs': 200,           # Väčší priestor pre učenie
    'batch_size': 64,        # Menšie dávky pre komplexnejšie vzorce
    'learning_rate': 0.00005 # Nižšia počiatočná rýchlosť učenia
    },
    'XAUUSD_ecn': {
        'lookback_window': 70,
        'threshold': 0.0016,
        'lstm_units': [104, 72],
        'risk_multiplier': 1.1
    },
    'DAX_ecn': {
        'lookback_window': 65,
        'threshold': 0.0014,
        'lstm_units': [96, 64],
        'risk_multiplier': 1.0
    },
    'NSDQ_ecn': {
        'lookback_window': 55,
        'threshold': 0.0013,
        'lstm_units': [88, 56],
        'risk_multiplier': 1.0
    },
    'SP_ecn': {
        'lookback_window': 60,
        'threshold': 0.0012,
        'lstm_units': [92, 60],
        'risk_multiplier': 1.0
    },
     'TSLA': {
        'lookback_window': 50,
        'threshold': 0.0025,  # Vyšší threshold kvôli vyššej volatilite
        'lstm_units': [128, 96],
        'risk_multiplier': 1.8  # Vyššie riziko pre volatilné akcie
    },
    'AAPL': {
        'lookback_window': 60,
        'threshold': 0.0012,
        'lstm_units': [96, 64],
        'risk_multiplier': 0.9  # Nižšie riziko pre stabilnejšiu akciu
    },
    'NVDA': {
        'lookback_window': 70,
        'threshold': 0.0022,
        'lstm_units': [120, 80],
        'risk_multiplier': 1.5  # Stredné riziko pre rýchlo rastúcu akciu
    }
}

def get_symbol_params(symbol):
    """Vráti špecifické parametre pre symbol alebo default"""
    default = MODEL_PARAMS.copy()
    return SYMBOL_HYPERPARAMS.get(symbol, default)

# Parametre pre online kalibráciu
ONLINE_CALIBRATION = {
    'calibration_interval': 100,  # Počet predikcií medzi kalibráciami
    'learning_rate': 0.0001,      # Learning rate pre online učenie
    'batch_size': 32,             # Veľkosť dávky pre kalibráciu
    'memory_size': 1000           # Počet posledných vzoriek pre kalibráciu
}

# Cesta k všeobecnému modelu pre transfer learning
GENERAL_MODEL_PATH = 'models/general_model.keras'