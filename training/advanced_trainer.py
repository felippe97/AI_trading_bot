# training/advanced_models.py
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import sys
import os
from datetime import datetime  # Added datetime import

# Absolute path to the project
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

# Import configuration
from config import get_symbol_params, GENERAL_MODEL_PATH, ONLINE_CALIBRATION

# Import feature engineering
from feature_engineering.advanced_features import AdvancedFeatureEngineer

class AdvancedModelTrainer:
    def __init__(self, symbol, model_type, data_path, timeframe='5min'):
        self.symbol = symbol
        self.model_type = model_type
        self.timeframe = timeframe
        self.data_path = data_path
        
        # Load symbol-specific parameters
        self.params = get_symbol_params(symbol)
        self.params['model_type'] = model_type
        print(f"Using parameters for {symbol}: {self.params}")
        
        self.feature_columns = []
        self.scalers = {}
        self.model = None
        self.history = None
        
        # Initialize feature engineer
        self.engineer = AdvancedFeatureEngineer(symbol, mode='training')
        
        # Load and preprocess data
        self.data = self._load_and_preprocess()
        
        # Prepare sequences
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_sequences()

    def _load_and_preprocess(self):
        """Load and preprocess data with advanced features"""
        print(f"Loading data for {self.symbol}...")
        df = pd.read_csv(self.data_path, parse_dates=['date'], index_col='date')
        
        # Add advanced features
        print("Adding advanced features...")
        df = self.engineer.add_all_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Add target variable
        df = self._create_target_variable()
        
        # Store feature columns
        self.feature_columns = [col for col in df.columns if col not in ['target']]
        
        return df
    
    def _handle_missing_values(self, df):
        """Advanced missing value handling"""
        # Forward fill for most features
        df[self.feature_columns] = df[self.feature_columns].ffill()
        
        # Backward fill any remaining NaNs
        df[self.feature_columns] = df[self.feature_columns].bfill()
        
        # Fill any remaining NaNs with 0
        df[self.feature_columns] = df[self.feature_columns].fillna(0)
        
        return df
    
    def _create_target_variable(self):
        """Create target variable based on future price movement"""
        # Calculate future returns using symbol-specific future_bars
        future_bars = self.params.get('future_bars', 3)
        threshold = self.params.get('threshold', 0.0015)
        
        self.data['future_close'] = self.data['close'].shift(-future_bars)
        self.data['future_return'] = (self.data['future_close'] - self.data['close']) / self.data['close']
        
        # Create multi-class target using symbol-specific threshold
        conditions = [
            self.data['future_return'] < -threshold,  # Down
            (self.data['future_return'] >= -threshold) & (self.data['future_return'] <= threshold),  # Neutral
            self.data['future_return'] > threshold  # Up
        ]
        choices = [0, 1, 2]
        self.data['target'] = np.select(conditions, choices, default=1)
        
        # Drop rows with missing targets
        self.data = self.data.dropna(subset=['target'])
        
        return self.data
    
    def _scale_features(self, df):
        """Apply appropriate scaling to different feature types"""
        # Create a copy to preserve original data
        scaled_df = df.copy()
        
        # Different scalers for different feature types
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        volatility_cols = [col for col in df.columns if 'volatility' in col or 'atr' in col]
        macro_cols = [col for col in df.columns if 'vix' in col or 'inventory' in col or 'reserves' in col]
        
        # Initialize scalers
        self.scalers['price'] = RobustScaler()
        self.scalers['volatility'] = StandardScaler()
        self.scalers['macro'] = MinMaxScaler(feature_range=(-1, 1))
        
        # Apply scaling
        if price_cols:
            scaled_df[price_cols] = self.scalers['price'].fit_transform(scaled_df[price_cols])
        
        if volatility_cols:
            scaled_df[volatility_cols] = self.scalers['volatility'].fit_transform(scaled_df[volatility_cols])
        
        if macro_cols:
            scaled_df[macro_cols] = self.scalers['macro'].fit_transform(scaled_df[macro_cols])
        
        return scaled_df
    
    def _prepare_sequences(self):
        """Create input sequences for the model"""
        print("Preparing sequences...")
        
        # Scale features
        scaled_df = self._scale_features(self.data)
        
        # Create sequences using symbol-specific lookback window
        sequence_length = self.params.get('lookback_window', 60)
        
        sequences = []
        targets = []
        
        for i in range(len(scaled_df) - sequence_length - 1):
            seq = scaled_df[self.feature_columns].iloc[i:i+sequence_length].values
            target = scaled_df['target'].iloc[i+sequence_length]
            sequences.append(seq)
            targets.append(target)
        
        # Convert to numpy arrays
        X = np.array(sequences)
        y = np.array(targets)
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Reshape for LSTM input (samples, time steps, features)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(self.feature_columns)))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], len(self.feature_columns)))
        
        # One-hot encode targets
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)
        
        print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
        print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_hybrid_model(self):
        """Create hybrid CNN-LSTM-Attention model using symbol-specific parameters"""
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        
        # Get model parameters
        lstm_units = self.params.get('lstm_units', [96, 64])
        dropout_rate = self.params.get('dropout_rate', 0.3)
        learning_rate = self.params.get('learning_rate', 0.001)
        
        inputs = Input(shape=input_shape)
        
        # CNN part for local patterns
        conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = MaxPooling1D(pool_size=2)(conv1)
        
        conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = MaxPooling1D(pool_size=2)(conv2)
        
        # LSTM part for sequential dependencies
        lstm_layer = conv2
        for i, units in enumerate(lstm_units):
            lstm_layer = Bidirectional(LSTM(units, return_sequences=(i < len(lstm_units)-1)))(lstm_layer)
            lstm_layer = LayerNormalization()(lstm_layer)
            lstm_layer = Dropout(dropout_rate)(lstm_layer)
        
        # Attention mechanism
        attention = MultiHeadAttention(num_heads=4, key_dim=64)(lstm_layer, lstm_layer)
        attention = LayerNormalization()(attention + lstm_layer)  # Residual connection
        
        # Combine outputs
        flat = Flatten()(attention)
        dense1 = Dense(256, activation='relu', kernel_regularizer=l1_l2(0.001, 0.001))(flat)
        dense1 = BatchNormalization()(dense1)
        dense1 = Dropout(dropout_rate + 0.1)(dense1)  # Slightly higher dropout
        
        outputs = Dense(3, activation='softmax')(dense1)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def create_transfer_model(self):
        """Create model with transfer learning from a base model"""
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        
        # Load base model
        base_model = load_model(GENERAL_MODEL_PATH)
        
        # Freeze base layers
        for layer in base_model.layers[:-4]:
            layer.trainable = False
        
        inputs = Input(shape=input_shape)
        x = base_model(inputs)
        
        # Get symbol-specific parameters
        dense_units = self.params.get('lstm_units', [96, 64])
        dropout_rate = self.params.get('dropout_rate', 0.3)
        learning_rate = self.params.get('learning_rate', 0.0005)
        
        # Add new layers
        for units in dense_units:
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        
        outputs = Dense(3, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def train(self):
        """Train the model with symbol-specific parameters"""
        # Create model based on type
        if self.model_type == 'transfer':
            print("Using transfer learning...")
            self.model = self.create_transfer_model()
        else:
            print("Creating new hybrid model...")
            self.model = self.create_hybrid_model()
        
        # Print model summary
        self.model.summary()
        
        # Get training parameters
        epochs = self.params.get('epochs', 100)
        batch_size = self.params.get('batch_size', 128)
        
        # Callbacks
        checkpoint_dir = f"models/{self.symbol}/{self.timeframe}/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=self.params.get('early_stopping_patience', 15),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.2, 
                patience=self.params.get('lr_patience', 5), 
                min_lr=1e-6
            ),
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, f"{self.model_type}_best.h5"),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Train model
        print(f"Training model for {epochs} epochs with batch size {batch_size}...")
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save(os.path.join(checkpoint_dir, f"{self.model_type}_final.h5"))
        
        # Save scalers
        joblib.dump(self.scalers, os.path.join(checkpoint_dir, f"{self.model_type}_scalers.pkl"))
        
        # Save training metadata
        self.save_training_metadata()
        
        # Evaluate model
        self.evaluate()
        
        return self.history
    
    def save_training_metadata(self):
        """Save training metadata and parameters"""
        metadata = {
            'symbol': self.symbol,
            'model_type': self.model_type,
            'timeframe': self.timeframe,
            'features': self.feature_columns,
            'parameters': self.params,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'final_accuracy': self.history.history['val_accuracy'][-1],
            'final_loss': self.history.history['val_loss'][-1],
            'sequence_length': self.params.get('lookback_window', 60),
            'future_bars': self.params.get('future_bars', 3),
            'threshold': self.params.get('threshold', 0.0015)
        }
        
        save_path = f"models/{self.symbol}/{self.timeframe}/"
        os.makedirs(save_path, exist_ok=True)
        
        joblib.dump(metadata, os.path.join(save_path, f"{self.model_type}_metadata.pkl"))
        print(f"Metadata saved to {save_path}")
    
    def evaluate(self):
        """Evaluate model performance"""
        if not self.model:
            print("Model not trained yet!")
            return
        
        print("Evaluating model...")
        # Evaluate on test set
        loss, accuracy, precision, recall = self.model.evaluate(
            self.X_test, self.y_test, verbose=0
        )
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        print(f"\nTest Evaluation:")
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        
        # Classification report
        y_pred = self.model.predict(self.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes, target_names=['Down', 'Neutral', 'Up']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true_classes, y_pred_classes))
    
    def online_calibrate(self, new_data, true_label):
        """Perform online calibration of the model"""
        if not hasattr(self, 'calibration_counter'):
            self.calibration_counter = 0
            self.calibration_data = []
        
        self.calibration_counter += 1
        self.calibration_data.append((new_data, true_label))
        
        # Check if it's time to calibrate
        if self.calibration_counter >= ONLINE_CALIBRATION['calibration_interval']:
            print("Performing online calibration...")
            self._update_model_with_new_data()
            self.calibration_counter = 0
    
    def _update_model_with_new_data(self):
        """Update model using accumulated calibration data"""
        # Prepare new data
        X_new = []
        y_new = []
        
        for data, label in self.calibration_data:
            # Preprocess and scale
            processed = self.engineer.add_all_features(data)
            for scaler_name, scaler in self.scalers.items():
                cols = [col for col in processed.columns if col in self.feature_columns]
                if cols:
                    processed[cols] = scaler.transform(processed[cols])
            
            # Create sequence
            sequence = processed[self.feature_columns].values[-self.X_train.shape[1]:]
            sequence = sequence.reshape((1, sequence.shape[0], sequence.shape[1]))
            X_new.append(sequence[0])
            y_new.append(label)
        
        X_new = np.array(X_new)
        y_new = tf.keras.utils.to_categorical(y_new, num_classes=3)
        
        # Fine-tune model
        self.model.fit(
            X_new, y_new,
            epochs=ONLINE_CALIBRATION.get('epochs', 5),
            batch_size=ONLINE_CALIBRATION.get('batch_size', 32),
            verbose=1
        )
        
        # Save updated model
        checkpoint_dir = f"models/{self.symbol}/{self.timeframe}/"
        self.model.save(os.path.join(checkpoint_dir, f"{self.model_type}_calibrated.h5"))
        print("Calibrated model saved")
        
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