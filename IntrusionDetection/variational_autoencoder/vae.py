import os
import sys
import warnings
warnings.filterwarnings("ignore")
import keras
import numpy as np
import tensorflow as tf


from tensorflow import keras
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BaseAnomalyModel import BaseAnomalyModel


def sampling(args):
    z_mean, z_log_var = args
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(z_mean))
    return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, x):
        z_mean, z_log_var, z = self.encoder(x)
        decoded = self.decoder(z)
        return decoded

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            mean, log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # MSE reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(data - reconstruction), axis=1)
            )
            # KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1)
            )
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

class VAEModel(BaseAnomalyModel):
    """Variational Autoencoder for anomaly detection."""

    def __init__(self, config_file: str = None):
        super().__init__("VAE", config_file)
        self.threshold = None
        self.history = None
        self.model = None

    def create_model(self) -> keras.Model:
        input_dim = self.config["input_dim"]
        latent_dim = self.config.get("latent_dim", 6)

        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        x = layers.Dense(self.config["units_first_layer"], activation=self.config["activation"])(encoder_input)
        x = layers.Dense(self.config["units_second_layer"], activation=self.config["activation"])(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = layers.Lambda(sampling)([z_mean, z_log_var])
        encoder = keras.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        decoder_input = layers.Input(shape=(latent_dim,))
        x = layers.Dense(self.config["units_second_layer"], activation=self.config["activation"])(decoder_input)
        x = layers.Dense(self.config["units_first_layer"], activation=self.config["activation"])(x)
        decoder_output = layers.Dense(input_dim, activation=None)(x)  # Linear output
        decoder = keras.Model(decoder_input, decoder_output, name="decoder")

        # Full VAE
        self.encoder = encoder
        self.decoder = decoder
        vae = VAE(encoder, decoder)
        vae.compile(optimizer=optimizers.Adam(learning_rate=self.config["learning_rate"]))
        return vae

    def train(self, x_train: np.ndarray, y_train: np.ndarray = None,
              x_val: np.ndarray = None, **kwargs) -> None:
        print(f"\n[II] Training {self.model_name} model...")
        tf.keras.utils.set_random_seed(42)
        self.history = self.model.fit(
            x_train,
            epochs=self.config["epochs"],
            batch_size=self.config["batch_size"],
            shuffle=True,
            verbose=1
        )

        if x_val is not None:
            self._calculate_threshold(x_val)

    def _calculate_threshold(self, x_val: np.ndarray) -> None:
        """Calculate anomaly threshold based on validation data reconstruction error."""
        _, _, z_val = self.encoder.predict(x_val, verbose=0)
        x_val_pred = self.decoder.predict(z_val, verbose=0)
        errors = np.mean(np.square(x_val - x_val_pred), axis=1)
        self.threshold = np.mean(errors) + 2 * np.std(errors)
        print(f"\nVariational AutoEncoder threshold: {self.threshold:.6f}")

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise ValueError("Model must be trained and threshold calculated before prediction")
        tf.keras.utils.set_random_seed(42)
        _, _, z_test = self.encoder.predict(x_test, verbose=0)
        x_test_pred = self.decoder.predict(z_test, verbose=0)
        errors_test = np.mean(np.square(x_test - x_test_pred), axis=1)
        y_pred = np.where(errors_test > self.threshold, -1, 1)
        return y_pred
    
    def get_anomaly_scores(self, x_test: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores based on reconstruction error.
        Higher values indicate more anomalous samples.
        """
        _, _, z_test = self.encoder.predict(x_test, verbose=0)
        x_test_pred = self.decoder.predict(z_test, verbose=0)
        errors_test = np.mean(np.square(x_test - x_test_pred), axis=1)
        return errors_test
    
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
    config_path = os.path.join(script_dir, 'vae_config.json')

    data_config_path = os.path.join(script_dir, '../config.json')
    
    model = VAEModel(config_path)
    model.run_pipeline(data_config_path, script_dir, scale=True)


if __name__ == "__main__":
    main()


        
