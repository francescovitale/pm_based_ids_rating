# Enhancing Anomaly-Based Intrusion Detection Systems with Process Mining

This project combines process mining with anomaly-based intrusion detection to rate alarms based on the degree of process-based severity obtained through process mining-driven analyses. 

The project is organized in two parts. 
- The first part - under the AnomalyDetection folder - includes the data sources and processing performed to train and test multiple machine learning models for detecting anomalies in network traffic flows.
- The second part - under the ProcessMining folder - includes the data sources and processing performed to train behavioral models against false positives resulted from the training process of intrusion detection, and the subsequent alarm rating at inference inference time.

Below are the descriptions of the two projects.

For issues, questions or contibutions, please contact [francesco.vitale@unina.it]-[francesco.grimaldi3@unina.it]-[fr.grimaldi@outlook.com].

## Anomaly Detection

### Project Structure

The project includes dedicated folders for each anomaly detection model:

- **`autoencoder/`**: Contains the Autoencoder model implementation
  - `auto_enc.py`: Autoencoder model class
  - `autoencoder_config.json`: Model-specific configuration
  - `results/`: Directory for storing model results (anomaly scores, false positives)

- **`ocsvm/`**: Contains the One-Class SVM model implementation
  - `ocsvm.py`: OCSVM model class
  - `ocsvm_config.json`: Model-specific configuration
  - `results/`: Directory for storing model results

- **`variational_autoencoder/`**: Contains the Variational Autoencoder model implementation
  - `vae.py`: VAE model class
  - `vae_config.json`: Model-specific configuration
  - `results/`: Directory for storing model results

Each model extends the `BaseAnomalyModel` abstract class, ensuring a consistent interface across different algorithms.

The **`dataset_flows/`** directory contains the network flow dataset organized as follows:

```
dataset_flows/
├── atk_flows/          # Attack/anomalous network flows (CSV files)
│   ├── slohttptestUV_1_flows.csv
│   ├── slowlorisRND_1_flows.csv
│   └── slowpostUV_1_flows.csv
├── nrml_flows/         # Normal/benign network flows (CSV files)
│   └── normal_1bisr_flows.csv
```

- **`atk_flows/`**: Contains CSV files with extracted features from attack traffic
- **`nrml_flows/`**: Contains CSV files with extracted features from normal traffic

- **`ModelRunner.py`**: Main orchestrator script that runs all configured models
- **`BaseAnomalyModel.py`**: Abstract base class defining the model interface
- **`DataManager.py`**: Handles data loading, preprocessing, and feature engineering
- **`ConfigManager.py`**: Manages configuration loading and validation
- **`ResultsManager.py`**: Handles results aggregation, storage, and reporting
- **`PMetrics.py`**: Provides performance metrics calculation (accuracy, precision, recall, F1-score, etc.)

- **`models_metrics_report.csv`**: Aggregated performance metrics for all models
- **`combined_false_positives.csv`**: Combined false positive analysis across models

### Configuration

**You must carefully modify the following configuration files before running the project:**

#### 1. `config.json` - Main Configuration

This file contains global settings for the framework:

```json
{
    "remove_features": ["flow_id", "timestamp", "src_ip", "dst_ip", "src_port", "dst_port"],
    "atk_flow_dir": "/path/to/dataset_flows/atk_flows",
    "nrml_flow_dir": "/path/to/dataset_flows/nrml_flows",
    "source_path": "/path/to/ids/"
}
```

**Key parameters to configure:**

- **`remove_features`**: List of features to exclude from training (non-predictive features)
- **`atk_flow_dir`**: Absolute path to the directory containing attack flow CSV files
- **`nrml_flow_dir`**: Absolute path to the directory containing normal flow CSV files
- **`source_path`**: Absolute path to the IDS project root directory
- **`models_list`**: List of model directories to execute (must match folder names)

#### 2. `models_config.json` - Models Configuration

This file defines which models to run and their basic parameters:

```json
{
    "models": [
        {
            "name": "OCSVM",
            "dir_name": "ocsvm",
            "class_name": "OCSVMModel",
            "config_file": "ocsvm_config.json",
            "params": {
                "scale": true
            }
        },
        {
            "name": "Autoencoder",
            "dir_name": "autoencoder",
            "class_name": "AutoencoderModel",
            "config_file": "autoencoder_config.json",
            "params": {
                "scale": true
            }
        },
        {
            "name": "VariationalAutoencoder",
            "dir_name": "variational_autoencoder",
            "class_name": "VAEModel",
            "config_file": "vae_config.json",
            "params": {
                "scale": true
            }
        }
    ]
}
```

#### 3. Model-Specific Configurations

Each model has its own configuration file (e.g., `autoencoder_config.json`, `ocsvm_config.json`, `vae_config.json`) located in the respective model directory. These files contain hyperparameters specific to each algorithm.

### Configuration Checklist

Before running the project, ensure:

- All paths in `config.json` are absolute and correctly point to your local directories
- `atk_flow_dir` and `nrml_flow_dir` point to the correct dataset locations
- `source_path` matches your IDS project root directory
- Models listed in `models_config.json` match the folders you want to execute
- Model-specific configuration files in each model folder are properly configured

### Usage

1. **Prepare the environment:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure the system:**
   - Edit `config.json` with your paths and feature settings
   - Edit `models_config.json` to select which models to run
   - Adjust model-specific configurations if needed

3. **Run the framework:**
   ```bash
   python3 ModelRunner.py
   ```

4. **Review results:**
   - Check `models_metrics_report.csv` for performance comparison
   - Review individual model results in their respective `results/` directories
   - Analyze false positives in `combined_false_positives.csv`

## Process Mining

### Project requirements

This project has been executed on a Windows 11 machine with Pytest 8.4.2 and Python 3.11.5. A few libraries have been used within Python modules. Among these, there are:

- pm4py 2.7.11.11
- scipy 1.11.2
- scikit-learn 1.3.0
- google-genai 1.45.0
- openai 2.7.2

Please note that the list above is not comprehensive and there could be other requirements for running the project.

### Project execution

The first script to execute is the `data_extraction.py` script within the DataExtraction folder. This script sets up the dataset to use in the process mining training and inference parts depending on the different machine learning models. In particular, since each machine learning model may output different false-positive network flows, the script arranges the output directories pertaining to the false positives based on the labels contained in `false_positives.csv` file. The false positives are randomly split into training and inference folders to account for the randomicity of the results.

As the datasets have been generated, the project can be run through the `experimentation.bat` DOS script, which runs the process mining pipeline as follows. 

First, the DOS script `process_mining_training.bat` executes `event_log_extraction.py`, which extracts state-wise event logs from the PCAP files associated with the training false-positive flows (please refer to https://github.com/francescovitale/pm_video_game_traffic_analysis for further info about the state extraction process). Next, the training state-wise false positives are processed through the `process_mining_training.py` script to build Petri nets associated with different network protocol states. In addition, the script extracts the average alignment sets of false-positive traces to use at inference time to evaluate the severity of the alarms.

Following the training phase is the execution of the DOS script `process_mining_inference.bat`. This script first executes the `event_log_extraction.py` to extract state-wise event logs from the PCAP files associated with the inference false-positive and true-positive flows. Next, event logs are compared to the state-wise Petri nets to extract the cosine similarity of the alignment distributions of the inference false-positive and true-positive flows with the alignment distribution of the training false-positive flows. 

The results of the execution of the `experimentation.bat` DOS script can be found under the Results folder, which contains, for each repetition of the experiment, the results associated with each machine learning models. In particular, the similarities for each inference true-positive and false-positive flows are contained in the `analysis_fp_similarities.txt` and `analysis_tp_similarities.txt` files, whereas more general analyses can be found in the `analysis_results.txt` file.