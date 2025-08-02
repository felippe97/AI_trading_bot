# training/hyperparameter_tuner.py
from symbol_trainer import IntradayModelTrainer
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from config import SYMBOLS

def create_model(optimizer='adam', neurons=64, dropout_rate=0.2):
    # Táto funkcia bude v skutočnosti definovaná v symbol_trainer
    pass

def tune_hyperparameters(symbol):
    trainer = IntradayModelTrainer(symbol, f"../data/{symbol}_M1.csv")
    X, y = trainer.prepare_dataset()
    
    model = KerasClassifier(build_fn=create_model, verbose=0)
    
    param_grid = {
        'batch_size': [64, 128],
        'epochs': [50, 100],
        'optimizer': ['adam', 'rmsprop'],
        'neurons': [64, 96],
        'dropout_rate': [0.2, 0.3]
    }
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X, y)
    
    print(f"Best parameters for {symbol}: {grid_result.best_params_}")
    return grid_result.best_params_

if __name__ == "__main__":
    from config import SYMBOLS
    for symbol in SYMBOLS:
        best_params = tune_hyperparameters(symbol)
        print(f"{symbol} best params: {best_params}")