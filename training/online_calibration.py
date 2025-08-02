# training/online_calibration.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import logging

logger = logging.getLogger(__name__)

class RealTimeCalibrator:
    def __init__(self, model_path, learning_rate=0.0001):
        self.model = tf.keras.models.load_model(model_path)
        self.original_model = tf.keras.models.clone_model(self.model)
        self.original_model.set_weights(self.model.get_weights())
        self.optimizer = Adam(learning_rate=learning_rate, clipvalue=0.5)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.calibration_losses = []
    
    def calibrate(self, X_batch, y_batch):
        """Vykoná jeden krok kalibrácie na nových dátach"""
        with tf.GradientTape() as tape:
            predictions = self.model(X_batch, training=True)
            loss = self.loss_fn(y_batch, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Uloženie histórie strát pre monitorovanie
        self.calibration_losses.append(loss.numpy())
        return loss.numpy()
    
    def revert_if_degraded(self, validation_data, threshold=0.1):
        """Vráti pôvodné váhy ak došlo k degradácii"""
        X_val, y_val = validation_data
        original_loss = self.loss_fn(y_val, self.original_model.predict(X_val))
        new_loss = self.loss_fn(y_val, self.model.predict(X_val))
        
        if new_loss > original_loss * (1 + threshold):
            self.model.set_weights(self.original_model.get_weights())
            logger.warning(f"Degradácia modelu ({new_loss:.4f} > {original_loss:.4f}). Obnovenie pôvodných váh.")
            return True
        return False
    
    def save_model(self, save_path):
        self.model.save(save_path)