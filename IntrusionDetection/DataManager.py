import os 
import json
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataManager: 
    train_path: str
    test_path: str

    def set_train_path(self, path: str) -> None:
        self.train_path = path

    def get_train_path(self) -> str:
        return self.train_path
    
    def set_test_path(self, path: str) -> None:
        self.test_path = path

    def get_test_path(self) -> str:
        return self.test_path
    
    def remove_flows(self, df: pd.DataFrame, flow_col: str) -> pd.DataFrame:
        """
        Remove rows where the flow column has a value of 0.
        Args:
            df (pd.DataFrame): Input DataFrame.
            flow_col (str): Name of the column representing flows.
        Returns:
            pd.DataFrame: DataFrame with rows having flow_col == 0 removed.
        """
        return df[df[flow_col] != 0].reset_index(drop=True)

    def delete_df_cols(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        return df.drop(columns=cols)

    def concat_df(self, df_list: list[pd.DataFrame]) -> pd.DataFrame:
        if len(df_list) == 0:
            raise ValueError("The provided list of DataFrames is empty.")
        else: 
            for df in df_list:
                self.order_df(df, col='timestamp')
            return pd.concat(df_list, ignore_index=True)

    def order_df(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        return df.sort_values(by=col).reset_index(drop=True)

    def check_dir(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"The specified path {path} does not exist.")
        
    def convert_features_to_numeric(self, df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, dict]:
        unique_labels = df[column].unique()
        label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}
        df[column] = df[column].map(label_to_numeric)
        return df, label_to_numeric

    def replace_non_numeric_with_nan(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Replace non-numeric values in a specified column with NaN.
        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Name of the column to clean.
        Returns:
            pd.DataFrame: DataFrame with non-numeric values replaced by NaN.
        """
        df[column] = pd.to_numeric(df[column], errors='coerce')
        return df
    
    def convert_label(self, df: pd.DataFrame, numeric: int): 
        "Convert the \'label\' column to a numeric value."
        df["label"] = numeric
        return df
        
    def load_flow(self, filename: str) -> pd.DataFrame:
        self.check_dir(filename)
        return pd.read_csv(filename)

    def load_flows(self, path: str) -> list[pd.DataFrame]:
        self.check_dir(path)
        all_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
        df_list = []
        for file in all_files:
            df = pd.read_csv(file)
            filename = os.path.splitext(os.path.basename(file))[0]
            # Create flow_identifier using existing 'index' column or DataFrame index
            if 'index' in df.columns:
                df['flow_identifier'] = df['index'].apply(lambda x: f"{filename}_flow_{x}")
            else:
                df['flow_identifier'] = df.index.to_series().apply(lambda x: f"{filename}_flow_{x}")
            df_list.append(df)
        return df_list
    
    def clean_infinite_and_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace infinite values with NaN and then fill NaN values with 0.
        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        return df
    
    def define_data(self, config: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        """
        Load, preprocess, and split the dataset into training and testing sets. 
        Normal flows are used as training set, while attack flows are used as testing set.
        The normal flows are store in the training directory (config["nrml_flow_dir"]) and
        the attack flows are stored in the testing directory (config["atk_flow_dir"]).

        Configuration is read from a JSON file and includes paths and features to remove.
        
        Args:
            config (str): Path to the configuration JSON file.
        
        Returns:
            tuple:
                - x_train (np.ndarray): Training feature set.
                - y_train (np.ndarray): Training labels.
                - x_test (np.ndarray): Testing feature set.
                - y_test (np.ndarray): Testing labels.
                - flow_id_train (np.ndarray): Flow identifiers for training set.
                - flow_id_test (np.ndarray): Flow identifiers for test set.
        """
        with open(config, 'r') as f:
            print("\n[I] Loading configuration...")
            config = json.load(f)

        self.set_train_path(config["nrml_flow_dir"])
        self.set_test_path(config["atk_flow_dir"])

        df_train_list = self.load_flows(self.get_train_path())
        df_test_list = self.load_flows(self.get_test_path())

        df_train = self.concat_df(df_train_list)
        df_test = self.concat_df(df_test_list)

        # Extract flow identifiers before any processing
        flow_id_train = df_train['flow_identifier'].values
        flow_id_test = df_test['flow_identifier'].values

        # drop useless flows (rows) 
        df_train = self.remove_flows(df_train, flow_col='duration')
        df_test = self.remove_flows(df_test, flow_col='duration')

        # Update flow identifiers after removing rows
        flow_id_train = df_train['flow_identifier'].values
        flow_id_test = df_test['flow_identifier'].values

        # drop features (columns), including flow_identifier
        features_to_remove = config['remove_features'] + ['flow_identifier']
        df_train = self.delete_df_cols(df_train, features_to_remove)
        df_test = self.delete_df_cols(df_test, features_to_remove)

        df_train, _ = self.convert_features_to_numeric(df_train, 'protocol')
        df_test, protocol_to_numeric = self.convert_features_to_numeric(df_test, 'protocol')
        print(f"[I] Protocol to numeric mapping: {protocol_to_numeric}")

        df_train = self.replace_non_numeric_with_nan(df_train, 'delta_start')
        df_test = self.replace_non_numeric_with_nan(df_test, 'delta_start')
        df_train = self.replace_non_numeric_with_nan(df_train, 'handshake_duration')
        df_test = self.replace_non_numeric_with_nan(df_test, 'handshake_duration')

        df_train = self.convert_label(df_train, 1)
        df_test = self.convert_label(df_test, -1)

        # Clean infinite and NaN values
        df_train = self.clean_infinite_and_nan(df_train)
        df_test = self.clean_infinite_and_nan(df_test)

        x_train = df_train.drop(columns=['label']).to_numpy()
        y_train = df_train['label'].to_numpy()

        x_test = df_test.drop(columns=['label']).to_numpy()
        y_test = df_test['label'].to_numpy()

        f.close()

        return x_train, y_train, x_test, y_test, flow_id_train, flow_id_test
    
    def data_scaling(self, x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Scale features to a given range using Min-Max scaling.
        Args:
            x_train (np.ndarray): Training feature set.
            x_test (np.ndarray): Testing feature set.
        Returns:
            tuple: 
                - x_train_scaled (np.ndarray): Scaled training feature set.
                - x_test_scaled (np.ndarray): Scaled testing feature set.   
        """
        scaler = MinMaxScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        return x_train_scaled, x_test_scaled
