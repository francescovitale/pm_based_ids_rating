from abc import ABC, abstractmethod
import os
import json
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Import from parent directory
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'anomaly_detection'))
from DataManager import DataManager
from PMetrics import Metrics


class BaseAnomalyModel(ABC):
    """Base class for all anomaly detection models."""
    
    def __init__(self, model_name: str, config_file: str = None):
        self.model_name = model_name
        self.model = None
        self.config = None
        self.metrics = Metrics()
        self.dm = DataManager()
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> Dict[str, Any]:
        """Load model-specific configuration."""
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        print(f"\n[I] Loaded {self.model_name} configuration")
        print(f"Parameters: {self.config}")
        return self.config
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create and return the model instance. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray = None, **kwargs) -> None:
        """
        Train the model. Must be implemented by subclasses.
        
        Args:
            x_train: Training features
            y_train: Training labels (optional, not used for unsupervised methods)
            **kwargs: Additional training parameters (e.g., validation data)
        """
        pass
    
    @abstractmethod
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Make predictions. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_anomaly_scores(self, x_test: np.ndarray) -> np.ndarray:
        """Calculate anomaly scores for each sample. Must be implemented by subclasses."""
        pass
    
    def prepare_data(self, config_path: str, 
                    validation_split: float = 0.4,
                    scale: bool = True) -> Tuple[np.ndarray, ...]:
        """
        Prepare training and test data.
        
        Args:
            config_path: Path to data configuration file
            validation_split: Percentage of training data to use for validation/test
            scale: Whether to scale the data
            
        Returns:
            Tuple of prepared data arrays
        """
        # Load data
        x_train, y_train, x_atk_test, y_atk_test, flow_id_train, flow_id_atk_test = \
            self.dm.define_data(config_path)
        
        # Scale if needed
        if scale:
            x_train, x_atk_test = self.dm.data_scaling(x_train, x_atk_test)
        
        # Split normal data into train and validation
        x_train, x_normal_test, y_train, y_normal_test, flow_id_train, flow_id_normal_test = \
            train_test_split(x_train, y_train, flow_id_train, 
                           test_size=validation_split, random_state=42)
        
        # Combine normal and attack test data
        x_test = np.vstack([x_normal_test, x_atk_test])
        y_test = np.concatenate([y_normal_test, y_atk_test])
        flow_id_test = np.concatenate([flow_id_normal_test, flow_id_atk_test])

        print(f"\n[prepare_data] Data shapes:")
        print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"  x_normal_test: {x_normal_test.shape}, y_normal_test: {y_normal_test.shape}")
        print(f"  x_atk_test: {x_atk_test.shape}, y_atk_test: {y_atk_test.shape}")
        print(f"  x_test: {x_test.shape}, y_test: {y_test.shape}")
        
        return (x_train, y_train, x_normal_test, y_normal_test, x_test, y_test,
                flow_id_train, flow_id_normal_test, flow_id_test,
                len(y_normal_test), len(y_atk_test))
    
    def evaluate(self, y_test: np.ndarray, y_pred: np.ndarray,
                flow_id_test: np.ndarray, benign_len: int, anomaly_len: int,
                script_dir: str, x_test: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate model predictions and save results.
        
        Args:
            y_test: True labels
            y_pred: Predicted labels
            flow_id_test: Flow identifiers
            benign_len: Number of benign samples
            anomaly_len: Number of anomaly samples
            script_dir: Directory to save results
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        
        print(f"\n[III] Evaluating {self.model_name} model...")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=-1, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=-1, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=-1, average='binary', zero_division=0)
        
        # Confusion matrix (inverted interpretation: attacks=-1 are positive class)
        cm = confusion_matrix(y_test, y_pred, labels=[1, -1])  # [normal, attack]
        tn, fp, fn, tp = cm.ravel()  # Now: tn/fp=normal, tp/fn=attack
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Display metrics
        self.metrics.show_metrics(y_test, y_pred)
        self.metrics.conf_matrix_values(y_test, y_pred, 
                                       y_val_len=benign_len, 
                                       y_test_len=anomaly_len)
        self.metrics.save_results(y_test=y_test, 
                                 y_pred=y_pred, 
                                 flow_id_test=flow_id_test,
                                 script_dir=script_dir,
                                 alg_name=self.model_name)
        
        # Save anomaly scores if x_test is provided
        if x_test is not None:
            self._save_anomaly_scores(x_test, flow_id_test, script_dir)
        
        # Return metrics dictionary
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'specificity': specificity,
            'fpr': false_positive_rate,
            'fnr': false_negative_rate,
            'benign_samples': benign_len,
            'anomaly_samples': anomaly_len
        }
    
    def run_pipeline(self, data_config_path: str, script_dir: str,
                    validation_split: float = 0.4, scale: bool = True,
                    **kwargs) -> Dict[str, float]:
        """
        Complete training and evaluation pipeline.
        
        Args:
            data_config_path: Path to data configuration
            script_dir: Script directory for saving results
            validation_split: Validation split ratio
            scale: Whether to scale data
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        print(f"\n----- Starting {self.model_name}...")
        
        # Prepare data
        (x_train, y_train, x_val, y_val, x_test, y_test,
         flow_id_train, flow_id_val, flow_id_test,
         benign_len, anomaly_len) = self.prepare_data(data_config_path, 
                                                       validation_split, scale)

        print(f"\n[run_pipeline] Data prepared. Training samples: {len(x_train)}, Test samples: {len(x_test)}")
        
        # Create and train model
        self.model = self.create_model()
        print(f"\n[II] Training {self.model_name} model...")
        self.train(x_train, y_train, **kwargs)
        
        # Predict and evaluate
        print("--------------------------------")
        y_pred = self.predict(x_test)
        metrics = self.evaluate(y_test, y_pred, flow_id_test, benign_len, anomaly_len, script_dir, x_test)
        
        print(f"\n----- {self.model_name} finished.")
        
        return metrics
    
    def _save_anomaly_scores(self, x_test: np.ndarray, flow_id_test: np.ndarray, 
                            script_dir: str) -> None:
        """Save anomaly scores to CSV file."""
        import pandas as pd
        
        print(f"\n[IV] Saving anomaly scores...")
        
        # Calculate anomaly scores
        scores = self.get_anomaly_scores(x_test)
        
        # Create DataFrame
        df = pd.DataFrame({
            'flow_id': flow_id_test,
            'anomaly_score': scores
        })
        
        # Create results directory if not exists
        results_dir = os.path.join(script_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Save to CSV
        output_path = os.path.join(results_dir, 'anomaly_scores.csv')
        df.to_csv(output_path, index=False)
        
        print(f"Anomaly scores saved to: {output_path}")
        print(f"Score statistics - Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
        print(f"                   Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")
