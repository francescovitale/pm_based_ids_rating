import os
import sys
import numpy as np
from sklearn import svm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BaseAnomalyModel import BaseAnomalyModel


class OCSVMModel(BaseAnomalyModel):
    """One-Class SVM implementation."""
    
    def __init__(self, config_file: str = None):
        super().__init__("OCSVM", config_file)
    
    def create_model(self) -> svm.OneClassSVM:
        """Create OCSVM model with configuration."""
        gamma = 1 / self.config["n_features"]
        
        model = svm.OneClassSVM(
            gamma=gamma,            # higher gamma -> ondulated decision boundary
            nu=self.config["nu"],   # higher nu -> more permissive to outliers in training data
            kernel=self.config["kernel"]  # radial kernel (non-linear) -> good for non-linear data
        )
        return model
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray = None, **kwargs) -> None:
        """Train the OCSVM model."""
        self.model.fit(x_train)
    
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(x_test)
    
    def get_anomaly_scores(self, x_test: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores using decision function.
        More negative values indicate more anomalous samples.
        """
        return self.model.decision_function(x_test)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'ocsvm_config.json')
    
    # Load data configuration
    data_config_path = os.path.join(script_dir, '../config.json')
    
    model = OCSVMModel(config_path)
    model.run_pipeline(data_config_path, script_dir, scale=True)


if __name__ == "__main__":
    main()
