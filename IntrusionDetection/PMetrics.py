import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score


class Metrics: 

    def save_results(self, y_test: np.ndarray, y_pred: np.ndarray, flow_id_test: np.ndarray, script_dir: str, alg_name: str):
        """

        Save false positives and false negatives to CSV files.
        Args:
            y_test (array-like): True labels.
            y_pred (array-like): Predicted labels.
            flow_id_test (array-like): Identifiers for each test instance.
            script_dir (str): Directory to save the results.
            alg_name (str): Name of the algorithm (used in column naming).
        
        """

        # Inverted interpretation: attack=-1 is positive, normal=1 is negative
        fp_mask = (y_test == 1) & (y_pred == -1)  # Normal classified as attack (FP)
        fn_mask = (y_test == -1) & (y_pred == 1)  # Attack classified as normal (FN)
        
        fp_identifiers = flow_id_test[fp_mask]
        fn_identifiers = flow_id_test[fn_mask]
        
        save_path = os.path.join(script_dir, 'results')
        os.makedirs(save_path, exist_ok=True)
        
        # Delete existing files before saving new ones
        fp_path = os.path.join(save_path, 'false_positives.csv')
        fn_path = os.path.join(save_path, 'false_negatives.csv')
        if os.path.exists(fp_path):
            os.remove(fp_path)
        if os.path.exists(fn_path):
            os.remove(fn_path)
        
        if len(fp_identifiers) > 0:
            fp_df = pd.DataFrame({alg_name + '_flowID': fp_identifiers})
            fp_df.to_csv(fp_path, index=False)
            print(f"\nSaved {len(fp_identifiers)} False Positives to false_positives.csv")
        
        if len(fn_identifiers) > 0:
            fn_df = pd.DataFrame({alg_name + '_flowID': fn_identifiers})
            fn_df.to_csv(fn_path, index=False)
            print(f"\nSaved {len(fn_identifiers)} False Negatives to false_negatives.csv")

    def conf_matrix_values(self, y_test, y_pred, y_val_len=None, y_test_len=None):
        # Confusion matrix (inverted interpretation: attacks=-1 are positive class)
        cm = confusion_matrix(y_test, y_pred, labels=[1, -1])  # [normal, attack]
        tn, fp, fn, tp = cm.ravel()  # Now: tn/fp=normal, tp/fn=attack
        
        print(f"\n[III] Confusion Matrix:")
        print(f"True Positives (TP - Attacks correctly detected): {tp} / {y_test_len}")
        print(f"True Negatives (TN - Normal correctly detected): {tn} / {y_val_len}")
        print(f"False Positives (FP - Normal classified as attack): {fp}")
        print(f"False Negatives (FN - Attack classified as normal): {fn}")
        

        # Calculate specificity and other metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0


        print(f"Specificity (TNR): {specificity:.4f}")
        print(f"False Positive Rate (FPR): {false_positive_rate:.4f}")
        print(f"False Negative Rate (FNR): {false_negative_rate:.4f}")

    def show_metrics(self, y_test, y_pred): 
        # Calculate metrics (attack=-1 as positive class)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=-1, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=-1, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=-1, average='binary', zero_division=0)
        
        
        print(f"\nMetrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity/TPR): {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        class_report = classification_report(y_test, y_pred)
        print(f"\nClassification Report:\n{class_report}")