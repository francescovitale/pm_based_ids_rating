import os
import pandas as pd
import numpy as np
from typing import List, Dict


class ResultsManager:
    """Manage and aggregate results from multiple models."""
    
    def __init__(self, source_path: str):
        self.source_path = source_path
    
    def load_model_results(self, model_name: str, result_type: str = 'false_positives') -> pd.DataFrame:
        """
        Load results for a specific model.
        
        Args:
            model_name: Name of the model
            result_type: Type of results ('false_positives' or 'false_negatives')
        """
        results_path = os.path.join(self.source_path, model_name, "results", f"{result_type}.csv")
        
        if os.path.exists(results_path):
            return pd.read_csv(results_path)
        return pd.DataFrame()
    
    def aggregate_results(self, models_list: List[str], result_type: str = 'false_positives') -> pd.DataFrame:
        """
        Aggregate results from multiple models.
        
        Args:
            models_list: List of model names
            result_type: Type of results to aggregate
        """
        all_ids = set()
        model_ids = {}
        
        for model in models_list:
            df = self.load_model_results(model, result_type)
            
            if not df.empty:
                id_column = df.columns[0]
                ids = set(df[id_column].values)
                model_ids[model] = ids
                all_ids.update(ids)
                print(f"Loaded {len(ids)} IDs from {model}")
        
        # Create aggregated DataFrame
        all_ids_list = sorted(list(all_ids))
        result_df = pd.DataFrame({'ID': all_ids_list})
        
        for model in models_list:
            result_df[model] = result_df['ID'].apply(
                lambda x: 1 if x in model_ids.get(model, set()) else 0
            )
        
        return result_df
    
    
    def find_common_ids(self, df: pd.DataFrame, models_list: List[str]) -> List:
        """Find IDs common to all models."""
        common_mask = df[models_list].sum(axis=1) == len(models_list)
        return df[common_mask]['ID'].tolist()
    
    def save_aggregated_results(self, df: pd.DataFrame, filename: str = 'combined_false_positives.csv') -> None:
        """Save aggregated results."""
        output_path = os.path.join(self.source_path, filename)
        df.to_csv(output_path, index=False)
        print(f"\nSaved combined results to: {output_path}")
