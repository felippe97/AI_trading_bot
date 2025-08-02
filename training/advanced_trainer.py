# training/advanced_models.py
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Conv1D, MaxPooling1D,
    Flatten, Bidirectional, BatchNormalization, concatenate,
    MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import logging
from datetime import datetime

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)
# Get the root directory (two levels up from the script)
project_root = os.path.dirname(os.path.dirname(current_script_path))
# Add root to Python path
sys.path.insert(0, project_root)

# Import configuration
from config import get_symbol_params, GENERAL_MODEL_PATH, ONLINE_CALIBRATION

# Import feature engineering
from feature_engineering.advanced_features import AdvancedFeatureEngineer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedModelTrainer:
    def __init__(self, symbol, model_type, data_path, timeframe='5min'):
        self.symbol = symbol
        self.model_type = model_type
        self.timeframe = timeframe
        
        # Convert relative path to absolute
        if not os.path.isabs(data_path):
            data_path = os.path.join(project_root, data_path)
        self.data_path = data_path
        
        # Load symbol-specific parameters
        self.params = get_symbol_params(symbol)
        self.params['model_type'] = model_type
        logger.info(f"Using parameters for {symbol}: {self.params}")
        
        self.feature_columns = []
        self.scalers = {}
        self.model = None
        self.history = None
        
        # Initialize feature engineer
        self.engineer = AdvancedFeatureEngineer(symbol, mode='training')
        
        # Load and preprocess data
        self.data = self._load_and_preprocess()
        
        # Prepare sequences
        self.X_seq_train, self.X_sent_train, self.X_seq_test, self.X_sent_test, self.y_train, self.y_test = self._prepare_sequences()

    def _load_and_preprocess(self):
        """Load and preprocess data with advanced features"""
        logger.info(f"Loading data for {self.symbol} from {self.data_path}")
        
        # Check if file exists
        if not os.path.exists(self.data_path):
            logger.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        try:
            df = pd.read_csv(self.data_path, parse_dates=['date'], index_col='date')
            
            # Validate data
            if df.empty:
                raise ValueError(f"Empty dataset: {self.data_path}")
            if 'close' not in df.columns:
                raise ValueError("Dataframe must contain 'close' column")
                
            logger.info(f"Loaded {len(df)} rows")
            
            # Add advanced features
            logger.info("Adding advanced features...")
            df = self.engineer.add_all_features(df)
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Add target variable
            df = self._create_target_variable(df)
            
            # Store feature columns
            self.feature_columns = [
                col for col in df.columns 
                if col not in ['target', 'future_close', 'future_return'] 
                and not col.startswith('sentiment')  # Sentiment bude spracovaný samostatne
            ]
            logger.info(f"Using features: {self.feature_columns}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _handle_missing_values(self, df):
        """Advanced missing value handling"""
        # Forward fill for most features
        df = df.ffill()
        
        # Backward fill any remaining NaNs
        df = df.bfill()
        
        # Fill any remaining NaNs with 0
        df = df.fillna(0)
        
        # Log missing values handling
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Still have {df.isnull().sum().sum()} missing values after cleaning")
        
        return df
    
    def _create_target_variable(self, df):
        """Create target variable based on future price movement"""
        # Calculate future returns using symbol-specific future_bars
        future_bars = self.params.get('future_bars', 3)
        threshold = self.params.get('threshold', 0.0015)
        
        df['future_close'] = df['close'].shift(-future_bars)
        df['future_return'] = (df['future_close'] - df['close']) / df['close']
        
        # Create multi-class target using symbol-specific threshold
        conditions = [
            df['future_return'] < -threshold,  # Down
            (df['future_return'] >= -threshold) & (df['future_return'] <= threshold),  # Neutral
            df['future_return'] > threshold  # Up
        ]
        choices = [0, 1, 2]
        df['target'] = np.select(conditions, choices, default=1)
        
        # Drop rows with missing targets
        initial_count = len(df)
        df = df.dropna(subset=['target'])
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} rows with missing targets")
        
        return df
    
    def _scale_features(self, df):
        """Apply appropriate scaling to different feature types"""
        # Create a copy to preserve original data
        scaled_df = df.copy()
        
        # Different scalers for different feature types
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        volatility_cols = [col for col in df.columns if 'volatility' in col or 'atr' in col]
        macro_cols = [col for col in df.columns if 'vix' in col or 'inventory' in col or 'reserves' in col]
        sentiment_cols = ['market_sentiment', 'news_sentiment']
        
        # Initialize scalers
        self.scalers['price'] = RobustScaler()
        self.scalers['volatility'] = StandardScaler()
        self.scalers['macro'] = MinMaxScaler(feature_range=(-1, 1))
        self.scalers['sentiment'] = MinMaxScaler(feature_range=(-1, 1))
        
        # Apply scaling
        if price_cols:
            scaled_df[price_cols] = self.scalers['price'].fit_transform(scaled_df[price_cols])
        
        if volatility_cols:
            scaled_df[volatility_cols] = self.scalers['volatility'].fit_transform(scaled_df[volatility_cols])
        
        if macro_cols:
            scaled_df[macro_cols] = self.scalers['macro'].fit_transform(scaled_df[macro_cols])
        
        if sentiment_cols:
            scaled_df[sentiment_cols] = self.scalers['sentiment'].fit_transform(scaled_df[sentiment_cols])
        
        return scaled_df
    
    def _prepare_sequences(self):
        """Create input sequences for the model with sentiment features"""
        logger.info("Preparing sequences...")
        
        # Scale features
        scaled_df = self._scale_features(self.data)
        
        # Create sequences using symbol-specific lookback window
        sequence_length = self.params.get('lookback_window', 60)
        
        sequences = []
        sentiment_sequences = []
        targets = []
        
        for i in range(len(scaled_df) - sequence_length - 1):
            # Time-series features
            seq = scaled_df[self.feature_columns].iloc[i:i+sequence_length].values
            
            # Sentiment features (only market and news sentiment)
            sent_seq = scaled_df[['market_sentiment', 'news_sentiment']].iloc[i:i+sequence_length].values
            
            # Target
            target = scaled_df['target'].iloc[i+sequence_length]
            
            sequences.append(seq)
            sentiment_sequences.append(sent_seq)
            targets.append(target)
        
        # Convert to numpy arrays
        X_seq = np.array(sequences)
        X_sent = np.array(sentiment_sequences)
        y = np.array(targets)
        
        # Split into train/test sets using time-based split
        split_index = int(len(X_seq) * 0.8)
        
        # Time-series features
        X_seq_train = X_seq[:split_index]
        X_seq_test = X_seq[split_index:]
        
        # Sentiment features
        X_sent_train = X_sent[:split_index]
        X_sent_test = X_sent[split_index:]
        
        # Targets
        y_train = y[:split_index]
        y_test = y[split_index:]
        
        # Reshape for LSTM input (samples, time steps, features)
        X_seq_train = X_seq_train.reshape((X_seq_train.shape[0], sequence_length, len(self.feature_columns)))
        X_seq_test = X_seq_test.reshape((X_seq_test.shape[0], sequence_length, len(self.feature_columns)))
        
        # Sentiment features are already in correct shape (samples, time steps, 2)
        
        # One-hot encode targets
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)
        
        logger.info(f"Train shapes: X_seq={X_seq_train.shape}, X_sent={X_sent_train.shape}, y={y_train.shape}")
        logger.info(f"Test shapes: X_seq={X_seq_test.shape}, X_sent={X_sent_test.shape}, y={y_test.shape}")
        
        return X_seq_train, X_sent_train, X_seq_test, X_sent_test, y_train, y_test
    
    def create_hybrid_model(self):
        """Create hybrid CNN-LSTM-Attention model with sentiment features"""
        # Input for time-series data
        time_series_input = Input(shape=(self.X_seq_train.shape[1], self.X_seq_train.shape[2]), name='time_series_input')
        
        # CNN part for local patterns
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(time_series_input)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        
        conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)
        
        # LSTM part for sequential dependencies
        lstm_layer = conv2
        lstm_units = self.params.get('lstm_units', [96, 64])
        for i, units in enumerate(lstm_units):
            lstm_layer = Bidirectional(LSTM(units, return_sequences=(i < len(lstm_units)-1)))(lstm_layer)
            lstm_layer = LayerNormalization()(lstm_layer)
            lstm_layer = Dropout(self.params.get('dropout_rate', 0.3))(lstm_layer)
        
        # Attention mechanism
        attention = MultiHeadAttention(
            num_heads=4, 
            key_dim=64,
            dropout=self.params.get('dropout_rate', 0.3)
        )(lstm_layer, lstm_layer)
        attention = LayerNormalization()(attention + lstm_layer)  # Residual connection
        
        # Flatten time-series branch
        time_flat = Flatten()(attention)
        
        # Input for sentiment data
        sentiment_input = Input(shape=(self.X_sent_train.shape[1], self.X_sent_train.shape[2]), name='sentiment_input')
        
        # Process sentiment with LSTM
        sent_lstm = LSTM(32, return_sequences=False)(sentiment_input)
        sent_lstm = Dropout(0.2)(sent_lstm)
        
        # Combine time-series and sentiment branches
        combined = concatenate([time_flat, sent_lstm])
        
        # Final dense layers
        dense1 = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(combined)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(self.params.get('dropout_rate', 0.3) + 0.1)(dense1)
        
        outputs = Dense(3, activation='softmax')(dense1)
        
        # Create model with two inputs
        model = Model(inputs=[time_series_input, sentiment_input], outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=self.params.get('learning_rate', 0.001)),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def create_transfer_model(self):
        """Create model with transfer learning from a base model"""
        # Load base model
        if not os.path.exists(GENERAL_MODEL_PATH):
            logger.warning(f"Base model not found at {GENERAL_MODEL_PATH}. Creating new hybrid model instead.")
            return self.create_hybrid_model()
            
        base_model = load_model(GENERAL_MODEL_PATH)
        
        # Freeze base layers
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        
        # Create new inputs
        time_series_input = Input(shape=(self.X_seq_train.shape[1], self.X_seq_train.shape[2]), name='time_series_input')
        sentiment_input = Input(shape=(self.X_sent_train.shape[1], self.X_sent_train.shape[2]), name='sentiment_input')
        
        # Pass through base model
        base_output = base_model(time_series_input)
        
        # Process sentiment with LSTM
        sent_lstm = LSTM(32, return_sequences=False)(sentiment_input)
        sent_lstm = Dropout(0.2)(sent_lstm)
        
        # Combine base output and sentiment
        combined = concatenate([base_output, sent_lstm])
        
        # Add new dense layers
        dense_units = self.params.get('lstm_units', [96, 64])
        dropout_rate = self.params.get('dropout_rate', 0.3)
        learning_rate = self.params.get('learning_rate', 0.0005)
        
        for units in dense_units:
            combined = Dense(units, activation='relu')(combined)
            combined = BatchNormalization()(combined)
            combined = Dropout(dropout_rate)(combined)
        
        outputs = Dense(3, activation='softmax')(combined)
        
        model = Model(inputs=[time_series_input, sentiment_input], outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def train(self):
        """Train the model with symbol-specific parameters"""
        # Create model based on type
        if self.model_type == 'transfer':
            logger.info("Using transfer learning...")
            self.model = self.create_transfer_model()
        else:
            logger.info("Creating new hybrid model...")
            self.model = self.create_hybrid_model()
        
        # Print model summary
        self.model.summary()
        
        # Get training parameters
        epochs = self.params.get('epochs', 100)
        batch_size = self.params.get('batch_size', 128)
        
        # Create checkpoint directory
        checkpoint_dir = os.path.join(project_root, f"models/{self.symbol}/{self.timeframe}/")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=self.params.get('early_stopping_patience', 15),
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.2, 
                patience=self.params.get('lr_patience', 5), 
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f"{self.model_type}_best.h5"),
                save_best_only=True,
                monitor='val_loss',
                save_weights_only=False,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(checkpoint_dir, 'logs'),
                histogram_freq=1
            )
        ]
        
        # Train model
        logger.info(f"Training model for {epochs} epochs with batch size {batch_size}...")
        self.history = self.model.fit(
            x=[self.X_seq_train, self.X_sent_train],
            y=self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=([self.X_seq_test, self.X_sent_test], self.y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(checkpoint_dir, f"{self.model_type}_final.h5")
        self.model.save(final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        
        # Save scalers
        scalers_path = os.path.join(checkpoint_dir, f"{self.model_type}_scalers.pkl")
        joblib.dump(self.scalers, scalers_path)
        logger.info(f"Saved scalers to {scalers_path}")
        
        # Save training metadata
        self.save_training_metadata()
        
        # Evaluate model
        self.evaluate()
        
        return self.history
    
    def save_training_metadata(self):
        """Save training metadata and parameters"""
        # Create directory if not exists
        save_path = os.path.join(project_root, f"models/{self.symbol}/{self.timeframe}/")
        os.makedirs(save_path, exist_ok=True)
        
        metadata = {
            'symbol': self.symbol,
            'model_type': self.model_type,
            'timeframe': self.timeframe,
            'features': self.feature_columns,
            'sentiment_features': ['market_sentiment', 'news_sentiment'],
            'parameters': self.params,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'final_accuracy': self.history.history['val_accuracy'][-1],
            'final_loss': self.history.history['val_loss'][-1],
            'sequence_length': self.params.get('lookback_window', 60),
            'future_bars': self.params.get('future_bars', 3),
            'threshold': self.params.get('threshold', 0.0015),
            'data_path': self.data_path
        }
        
        metadata_path = os.path.join(save_path, f"{self.model_type}_metadata.pkl")
        joblib.dump(metadata, metadata_path)
        logger.info(f"Metadata saved to {metadata_path}")
    
    def evaluate(self):
        """Evaluate model performance"""
        if not self.model:
            logger.error("Model not trained yet!")
            return
        
        logger.info("Evaluating model...")
        # Evaluate on test set
        results = self.model.evaluate(
            x=[self.X_seq_test, self.X_sent_test],
            y=self.y_test,
            verbose=0
        )
        metrics = dict(zip(self.model.metrics_names, results))
        
        f1 = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-7)
        
        logger.info("\nTest Evaluation:")
        logger.info(f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {f1:.4f}")
        
        # Classification report
        y_pred = self.model.predict(
            [self.X_seq_test, self.X_sent_test],
            verbose=0
        )
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_true_classes, y_pred_classes, target_names=['Down', 'Neutral', 'Up']))
        
        logger.info("\nConfusion Matrix:")
        logger.info(confusion_matrix(y_true_classes, y_pred_classes))
    
    def online_calibrate(self, new_data, true_label):
        """Perform online calibration of the model"""
        if not hasattr(self, 'calibration_counter'):
            self.calibration_counter = 0
            self.calibration_data = []
        
        self.calibration_counter += 1
        self.calibration_data.append((new_data, true_label))
        
        # Check if it's time to calibrate
        if self.calibration_counter >= ONLINE_CALIBRATION['calibration_interval']:
            logger.info("Performing online calibration...")
            self._update_model_with_new_data()
            self.calibration_counter = 0
    
    def _update_model_with_new_data(self):
        """Update model using accumulated calibration data"""
        # Prepare new data
        X_seq_new = []
        X_sent_new = []
        y_new = []
        
        for data, label in self.calibration_data:
            # Preprocess and scale
            processed = self.engineer.add_all_features(data)
            for scaler_name, scaler in self.scalers.items():
                cols = [col for col in processed.columns if col in self.feature_columns]
                if cols:
                    processed[cols] = scaler.transform(processed[cols])
            
            # Create sequences
            sequence_length = self.params.get('lookback_window', 60)
            
            # Time-series sequence
            ts_seq = processed[self.feature_columns].values[-sequence_length:]
            ts_seq = ts_seq.reshape((1, sequence_length, len(self.feature_columns)))
            
            # Sentiment sequence
            sent_seq = processed[['market_sentiment', 'news_sentiment']].values[-sequence_length:]
            sent_seq = sent_seq.reshape((1, sequence_length, 2))
            
            X_seq_new.append(ts_seq[0])
            X_sent_new.append(sent_seq[0])
            y_new.append(label)
        
        X_seq_new = np.array(X_seq_new)
        X_sent_new = np.array(X_sent_new)
        y_new = tf.keras.utils.to_categorical(y_new, num_classes=3)
        
        # Fine-tune model
        self.model.fit(
            x=[X_seq_new, X_sent_new],
            y=y_new,
            epochs=ONLINE_CALIBRATION.get('epochs', 5),
            batch_size=ONLINE_CALIBRATION.get('batch_size', 32),
            verbose=1
        )
        
        # Save updated model
        checkpoint_dir = os.path.join(project_root, f"models/{self.symbol}/{self.timeframe}/")
        calibrated_path = os.path.join(checkpoint_dir, f"{self.model_type}_calibrated.h5")
        self.model.save(calibrated_path)
        logger.info(f"Calibrated model saved to {calibrated_path}")
        
        # Clear calibration data
        self.calibration_data = []

# Example usage
if __name__ == "__main__":
    # Initialize trainer
    trainer = AdvancedModelTrainer(
        symbol="BTCUSD_ecn",
        model_type="hybrid",
        data_path="data/BTCUSD_ecn_5min.csv",
        timeframe="5min"
    )
    
    # Train model
    trainer.train()