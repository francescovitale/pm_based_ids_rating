import os
import sys
import pandas as pd
from typing import List, Dict

# Import all model classes
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ocsvm.ocsvm import OCSVMModel
from autoencoder.auto_enc import AutoencoderModel
from variational_autoencoder.vae import VAEModel
from ResultsManager import ResultsManager
from ConfigManager import ConfigManager


class ModelRunner:
    """Orchestrates running multiple anomaly detection models."""
    
    def __init__(self, base_config_path: str):
        self.base_config = ConfigManager.load_config(base_config_path)
        self.models_config = self._prepare_models_config()
        self.results = {}
        self.metrics_results = {}  # Store metrics for each model
    
    def _prepare_models_config(self) -> List[Dict]:
        """Prepare configuration for each model."""
        source_path = self.base_config['source_path']
        
        # Load models configuration from JSON
        models_config_path = os.path.join(os.path.dirname(__file__), 'models_config.json')
        models_json = ConfigManager.load_config(models_config_path)
        
        # Map class names to actual classes
        class_map = {
            'OCSVMModel': OCSVMModel,
            'AutoencoderModel': AutoencoderModel,
            'VAEModel': VAEModel
        }
        
        # Build config with full paths and class references
        models_config = []
        for model_def in models_json['models']:
            models_config.append({
                'name': model_def['name'],
                'dir_name': model_def['dir_name'],
                'class': class_map[model_def['class_name']],
                'config_file': os.path.join(source_path, model_def['dir_name'], model_def['config_file']),
                'script_dir': os.path.join(source_path, model_def['dir_name']),
                'params': model_def['params']
            })
        
        return models_config
    
    def run_single_model(self, model_config: Dict) -> None:
        """Run a single model."""
        try:
            print(f"\n{'='*60}")
            print(f"Running {model_config['name']}")
            print(f"{'='*60}")
            
            # Instantiate model
            model = model_config['class'](model_config['config_file'])
            
            # Run pipeline
            data_config_path = os.path.join(
                os.path.dirname(model_config['script_dir']), 
                'config.json'
            )
            
            metrics = model.run_pipeline(
                data_config_path=data_config_path,
                script_dir=model_config['script_dir'],
                **model_config['params']
            )
            
            self.results[model_config['name']] = 'Success'
            self.metrics_results[model_config['name']] = metrics
            
        except Exception as e:
            print(f"\n[ERROR] Failed to run {model_config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.results[model_config['name']] = f'Failed: {str(e)}'
    
    def run_all_models(self) -> None:
        """Run all configured models."""
        print("\n" + "="*60)
        print("STARTING ANOMALY DETECTION MODEL COMPARISON")
        print("="*60)
        
        for model_config in self.models_config:
            self.run_single_model(model_config)
        
        self._print_summary()
        self.save_metrics_report() 
    
    def _print_summary(self) -> None:
        """Print summary of all model runs."""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
    
    def save_metrics_report(self) -> None:
        """Save comprehensive metrics report to CSV."""
        if not self.metrics_results:
            print("No metrics to save.")
            return
        
        print("\n" + "="*60)
        print("SAVING METRICS REPORT")
        print("="*60)
        
        # Prepare data for DataFrame
        report_data = []
        for model_name, metrics in self.metrics_results.items():
            # Skip models that failed (metrics is None)
            if metrics is None:
                print(f"Skipping {model_name} - no metrics available")
                continue
                
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'Specificity': metrics['specificity'],
                'FPR': metrics['fpr'],
                'FNR': metrics['fnr'],
                'True Positives': metrics['tp'],
                'True Negatives': metrics['tn'],
                'False Positives': metrics['fp'],
                'False Negatives': metrics['fn'],
                'Benign Samples': metrics['benign_samples'],
                'Anomaly Samples': metrics['anomaly_samples']
            }
            report_data.append(row)
        
        # Check if we have any data to save
        if not report_data:
            print("No successful models to report.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(report_data)
        
        # Sort by F1-Score descending
        df = df.sort_values('F1-Score', ascending=False)
        
        # Save to CSV
        output_path = os.path.join(self.base_config['source_path'], 'models_metrics_report.csv')
        df.to_csv(output_path, index=False, float_format='%.4f')
        
        print(f"\nMetrics report saved to: {output_path}")
        print("\nMetrics Summary:")
        print(df.to_string(index=False))
        
    def aggregate_results(self) -> None:
        """Aggregate results from all models."""
        print("\n" + "="*60)
        print("AGGREGATING RESULTS")
        print("="*60)
        
        # Delete existing combined files before aggregating new results
        combined_fp_path = os.path.join(self.base_config['source_path'], 'combined_false_positives.csv')
        combined_fn_path = os.path.join(self.base_config['source_path'], 'combined_false_negatives.csv')
        if os.path.exists(combined_fp_path):
            os.remove(combined_fp_path)
        if os.path.exists(combined_fn_path):
            os.remove(combined_fn_path)
        
        rm = ResultsManager(self.base_config['source_path'])
        
        # Map model names to directory names for successful models
        model_dir_map = {config['name']: config['dir_name'] for config in self.models_config}
        
        # Get list of successfully run models (using directory names)
        successful_models = [
            model_dir_map[name] for name, status in self.results.items() 
            if status == 'Success'
        ]
        
        if not successful_models:
            print("No successful models to aggregate.")
            return
        if not successful_models:
            print("No successful models to aggregate.")
            return
        
        # Aggregate false positives
        print("\nAggregating False Positives...")
        fp_df = rm.aggregate_results(successful_models, 'false_positives')
        
        if not fp_df.empty:
            rm.save_aggregated_results(fp_df, 'combined_false_positives.csv')
            
            # Find common IDs
            common_fps = rm.find_common_ids(fp_df, successful_models)
            
            print(f"\nSummary:")
            for model in successful_models:
                count = fp_df[model].sum()
                print(f"  {model}: {count} false positives")
            
            print(f"\nCommon false positives to all models: {len(common_fps)}")
            if common_fps:
                print("IDs:")
                for fp_id in common_fps[:10]:  # Show first 10
                    print(f"  - {fp_id}")
                if len(common_fps) > 10:
                    print(f"  ... and {len(common_fps) - 10} more")
        
        # Aggregate false negatives
        print("\n\nAggregating False Negatives...")
        fn_df = rm.aggregate_results(successful_models, 'false_negatives')
        
        if not fn_df.empty:
            rm.save_aggregated_results(fn_df, 'combined_false_negatives.csv')
            
            common_fns = rm.find_common_ids(fn_df, successful_models)
            
            print(f"\nSummary:")
            for model in successful_models:
                count = fn_df[model].sum()
                print(f"  {model}: {count} false negatives")
            
            print(f"\nCommon false negatives to all models: {len(common_fns)}")


def main():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')
    
    runner = ModelRunner(config_path)
    runner.run_all_models()
    runner.aggregate_results()


if __name__ == "__main__":
    main()
