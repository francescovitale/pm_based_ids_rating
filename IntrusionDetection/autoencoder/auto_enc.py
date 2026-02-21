import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BaseAnomalyModel import BaseAnomalyModel

class AutoencoderModel(BaseAnomalyModel):
    """Autoencoder implementation for anomaly detection."""
    
    def __init__(self, config_file: str = None):
        super().__init__("AE", config_file)
        self.threshold = None
        self.history = None
    
    def create_model(self) -> models.Model:
        """Create Autoencoder model."""
        input_dim = self.config["input_dim"]
        
        # Encoder
        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(
            self.config["units_first_layer"], 
            activation=self.config["activation"]
        )(input_layer)
        encoded = layers.Dense(
            self.config["units_second_layer"], 
            activation=self.config["activation"]
        )(encoded)
        
        # Decoder
        decoded = layers.Dense(
            self.config["units_second_layer"], 
            activation=self.config["activation"]
        )(encoded)
        
        decoded = layers.Dense(
            self.config["units_first_layer"], 
            activation=self.config["activation"]
        )(decoded)
        decoded = layers.Dense(
            input_dim, 
            activation=self.config["activation"]
        )(decoded)
        
        # Autoencoder Model
        autoencoder = models.Model(inputs=input_layer, outputs=decoded)
        
        autoencoder.compile(
            optimizer=optimizers.Adam(learning_rate=self.config["learning_rate"]),
            loss=self.config["loss"]
        )
        
        return autoencoder
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray = None, 
              x_val: np.ndarray = None, **kwargs) -> None:
        """
        Train the Autoencoder model.
        
        Args:
            x_train: Training data
            y_train: Not used for autoencoder (kept for compatibility)
            x_val: Validation data for threshold calculation
        """
        print(f"\n[II] Training {self.model_name} model...")
        
        validation_data = (x_val, x_val) if x_val is not None else None
        tf.keras.utils.set_random_seed(42)
        self.history = self.model.fit(
            x_train, x_train,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            shuffle=True,
            validation_data=validation_data,
            verbose=1
        )
        
        if x_val is not None:
            self._calculate_threshold(x_val)
    
    def _calculate_threshold(self, x_val: np.ndarray) -> None:
        """Calculate anomaly threshold based on validation data reconstruction error."""
        val_recon = self.model.predict(x_val, verbose=0)
        mse = np.mean(np.power(x_val - val_recon, 2), axis=1)
        
        print(f"\n[III] Validation MSE: {np.mean(mse):.6f}")
        
        self.threshold = np.mean(mse) + 1.75 * np.std(mse)
        print(f"AutoEncoder Threshold: {self.threshold:.6f}")
    
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Make predictions based on reconstruction error.
        
        Returns:
            Predictions: -1 for anomaly, 1 for normal
        """
        if self.threshold is None:
            raise ValueError("Model must be trained and threshold calculated before prediction")
        tf.keras.utils.set_random_seed(42)
        test_recon = self.model.predict(x_test, verbose=0)
        mse_test = np.mean(np.power(x_test - test_recon, 2), axis=1)
        
        print(f"\n[IV] Test MSE: {np.mean(mse_test):.6f}")
        
        # -1 = anomaly, 1 = normal
        y_pred = np.where(mse_test > self.threshold, -1, 1)
        
        return y_pred
    
    def get_anomaly_scores(self, x_test: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores based on reconstruction error (MSE).
        Higher values indicate more anomalous samples.
        """
        test_recon = self.model.predict(x_test, verbose=0)
        mse_test = np.mean(np.power(x_test - test_recon, 2), axis=1)
        return mse_test
    
    def prepare_data(self, config_path: str, 
                    validation_split: float = 0.4,
                    internal_val_split: float = 0.14,
                    scale: bool = True):
        """
        Prepare data with additional internal validation split for threshold calculation.
        
        Args:
            config_path: Path to data configuration
            validation_split: Test split from normal data
            internal_val_split: Additional validation split for threshold
            scale: Whether to scale data
        """
        # Load data
        x_train, y_train, x_atk_test, y_atk_test, flow_id_train, flow_id_atk_test = \
            self.dm.define_data(config_path)
        
        # Scale if needed
        if scale:
            x_train, x_atk_test = self.dm.data_scaling(x_train, x_atk_test)
        
        # First split: train + internal_val vs normal_test
        x_train, x_normal_test, y_train, y_normal_test, flow_id_train, flow_id_normal_test = \
            train_test_split(x_train, y_train, flow_id_train, 
                           test_size=validation_split, random_state=42)
        
        # Second split: train vs internal validation (for threshold)
        x_train, x_val, y_train, y_val, flow_id_train, flow_id_val = \
            train_test_split(x_train, y_train, flow_id_train,
                           test_size=internal_val_split, random_state=42)
        
        # Combine normal and attack test data
        x_test = np.vstack([x_normal_test, x_atk_test])
        y_test = np.concatenate([y_normal_test, y_atk_test])
        flow_id_test = np.concatenate([flow_id_normal_test, flow_id_atk_test])

        print(f"\n[prepare_data] Data shapes:")
        print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"  x_val: {x_val.shape}, y_val: {y_val.shape}")
        print(f"  x_normal_test: {x_normal_test.shape}, y_normal_test: {y_normal_test.shape}")
        print(f"  x_atk_test: {x_atk_test.shape}, y_atk_test: {y_atk_test.shape}")
        print(f"  x_test: {x_test.shape}, y_test: {y_test.shape}")
        
        return (x_train, y_train, x_val, y_val, x_test, y_test,
                flow_id_train, flow_id_val, flow_id_test,
                len(y_normal_test), len(y_atk_test))
    
    def run_pipeline(self, data_config_path: str, script_dir: str,
                    validation_split: float = 0.4, 
                    internal_val_split: float = 0.14,
                    scale: bool = True):
        """
        Complete training and evaluation pipeline for Autoencoder.
        
        Args:
            data_config_path: Path to data configuration
            script_dir: Script directory for saving results
            validation_split: Test split ratio
            internal_val_split: Internal validation split for threshold
            scale: Whether to scale data
        """
        print(f"\n----- Starting {self.model_name}...")
        
        # Prepare data with internal validation split
        (x_train, y_train, x_val, y_val, x_test, y_test,
         flow_id_train, flow_id_val, flow_id_test,
         benign_len, anomaly_len) = self.prepare_data(
             data_config_path, validation_split, internal_val_split, scale
         )
        
        # Update config with actual input dimension
        self.config["input_dim"] = x_train.shape[1]
        
        # Create and train model
        self.model = self.create_model()
        self.train(x_train, y_train, x_val=x_val)
        
        print("--------------------------------")
        
        # Predict and evaluate
        print(f"\n[IV] Evaluating {self.model_name} model on test data...")
        y_pred = self.predict(x_test)
        
        print(f"\n[IV] Test Results:")
        metrics = self.evaluate(y_test, y_pred, flow_id_test, benign_len, anomaly_len, script_dir, x_test)
        
        print(f"\n----- {self.model_name} finished.")
        
        return metrics


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'autoencoder_config.json')

    data_config_path = os.path.join(script_dir, '../config.json')
    
    model = AutoencoderModel(config_path)
    model.run_pipeline(data_config_path, script_dir, scale=True)


if __name__ == "__main__":
    main()
