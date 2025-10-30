# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains research code for the EVSE (Electric Vehicle Supply Equipment) Dataset 2024 from the University of New Brunswick. The project focuses on cybersecurity attack detection for EVSE systems using machine learning and federated learning approaches.

**Key Citation**: Dataset from [UNB CIC Datasets](https://www.unb.ca/cic/datasets/evse-dataset-2024.html)

## Development Environment

### Package Management
- Uses `uv` as the package manager (not pip)
- Python 3.13+ required
- Dependencies defined in `pyproject.toml`

### Common Commands

```bash
# Install dependencies
uv sync

# Run data preprocessing and analysis
python data.py

# Run model training and evaluation
python model.py
```

## Code Architecture

### Two-Phase Pipeline

The codebase is structured as a two-phase ML pipeline:

#### Phase 1: Data Processing (`data.py`)
- **Main Class**: `EVSEDataProcessor`
- **Purpose**: Loads, cleans, and prepares EVSE network traffic data for ML
- **Key Responsibilities**:
  - Loads CSV data from `data/Host Events/EVSE-B-HPC-Kernel-Events-Combined.csv`
  - Performs data cleaning (outlier removal, missing values, duplicate columns)
  - Creates three balanced datasets:
    1. `evse_binary_classification.csv` - Attack vs Benign (binary)
    2. `evse_multiclass_attacks.csv` - Attack type classification (Cryptojacking, DoS, Reconnaissance)
    3. `evse_scenario_classification.csv` - Scenario-based classification
  - Generates comprehensive visualizations saved to `data-results/`
  - Uses SMOTE, ADASYN, and RandomOverSampler for class balancing

**Data Flow**: Raw CSV → Cleaning → Feature Engineering → Balanced Datasets → Visualizations

#### Phase 2: Model Training (`model.py`)
- **Main Class**: `NetworkTrafficDataProcessor` and `AdvancedTCN`
- **Purpose**: Trains deep learning models for attack detection
- **Key Components**:
  - **AdvancedTCN**: Temporal Convolutional Network with multi-head attention
  - **FederatedClient**: Implements federated learning for distributed training
  - Supports both federated and centralized training modes
  - Creates temporal sequences from preprocessed data
  - Includes Isolation Forest for anomaly detection
  - Generates evaluation visualizations in `model-results/`

**Model Architecture**: TCN blocks → Multi-head Attention → Global Pooling → Fully Connected Classifier

### Attack Type Mappings

The code uses these attack type mappings (from data.py:152-180):
```python
attack_mapping = {
    "none": "Benign",
    "cryptojacking": "Cryptojacking",
    "aggressive-scan": "Reconnaissance",
    "port-scan": "Reconnaissance",
    "service-detection": "Reconnaissance",
    "vuln-scan": "Reconnaissance",
    "icmp-flood": "DoS",
    "syn-flood": "DoS",
    "tcp-flood": "DoS",
    "udp-flood": "DoS",
    # ... (see full mapping in data.py)
}

scenario_mapping = {
    "Benign": "Benign",
    "Cryptojacking": "Cryptojacking",
    "Recon": "Reconnaissance",
    "DoS": "DoS",
}
```

### Visualization System

Both scripts generate extensive Plotly visualizations:
- **Global Layout** defined in `LAYOUT` variable (both files)
- **Color Palette** in `COLORS` array - use for consistency
- All plots saved as PNG using kaleido
- Results cleaned at each run (`data-results/` and `model-results/` directories are recreated)

### Key Design Patterns

1. **Results Folder Management**: Both scripts clear and recreate their results folders at startup (data.py:28-34, model.py:73-80)
2. **Sequence Creation**: Model training requires temporal sequences (default length=30) from the preprocessed data
3. **Multi-Target Support**: Code handles binary, multiclass (attack type), and scenario classification
4. **Device Flexibility**: PyTorch code automatically uses CUDA if available, otherwise CPU

## Important Implementation Details

### Data Preprocessing Validation
- The `NetworkTrafficDataProcessor.preprocess_data()` method has extensive error handling and validation
- Features with constant values, >95% missing data, or zero variance are automatically removed
- Data quality metrics are logged throughout preprocessing
- Target columns must be: `Label` (binary), `Attack` (multiclass), or `Scenario` (scenario)

### Model Training Modes
When running `model.py`, the script can operate in two modes:
1. **Federated**: Data split across multiple clients, models aggregated via `federated_average()`
2. **Centralized**: Standard single-model training

The `run_analysis()` function coordinates the entire training pipeline (model.py:1351).

### Logging
Both scripts use Python's logging module:
- `data.py`: Prints to console with emoji indicators
- `model.py`: Logs to both console and `model-results/analysis_log.txt`

## Dataset Structure

The EVSE dataset includes:
- Network traffic features (packet sizes, durations, protocols)
- Target variables: `State`, `Attack`, `Scenario`, `Label`
- Excluded columns: `interface`, `time`, empty columns

## Common Pitfalls

1. **Do not manually manage results folders** - both scripts auto-clean them
2. **Ensure CSV files exist** before running model training (run `data.py` first)
3. **Memory intensive**: Full dataset processing requires significant RAM
4. **GPU recommended** for model training but not required
5. **Sequence length** must be less than dataset size (auto-adjusted if needed)
