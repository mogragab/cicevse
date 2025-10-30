# data.py Code Review and Improvement Recommendations

## Executive Summary

The `data.py` file is well-structured with comprehensive preprocessing, visualization, and data balancing capabilities. However, several enhancements would improve alignment with the federated learning framework, reproducibility for the research paper, and overall code quality.

---

## Current Strengths ✅

1. **Comprehensive preprocessing pipeline** with 5 distinct phases
2. **Multiple data balancing strategies** (SMOTE, ADASYN, Random Oversampling)
3. **Extensive visualization suite** (18 different figures)
4. **Attack mapping standardization** to 4 categories (Benign, Cryptojacking, DoS, Reconnaissance)
5. **Baseline model evaluation** with Random Forest
6. **Clean class structure** with `EVSEDataProcessor`

---

## Recommended Improvements

### 1. **CRITICAL: Output Directory Naming**

**Issue**: Line 3 and 28-34
```python
data_results = 'data-results'  # Line 3 - defined but not used consistently
if os.path.exists("data-results"):  # Line 28 - hardcoded
```

**Problem**: The variable `data_results` is defined but not used throughout the code. All paths use hardcoded `"data-results"`.

**Recommendation**:
```python
DATA_RESULTS_DIR = 'data-results'  # Use constant naming convention

# Replace all occurrences of "data-results" with DATA_RESULTS_DIR
if os.path.exists(DATA_RESULTS_DIR):
    shutil.rmtree(DATA_RESULTS_DIR)
os.makedirs(DATA_RESULTS_DIR)

# In save functions:
fig.write_image(f"{DATA_RESULTS_DIR}/binary_distribution.png")
```

**Impact**: Improves maintainability and consistency.

---

### 2. **IMPORTANT: Output File Naming for Paper**

**Issue**: Lines 877, 901, 919
```python
df_binary.to_csv("evse_binary_classification.csv", index=False)
df_attack.to_csv("evse_multiclass_attacks.csv", index=False)
df_scenario.to_csv("evse_scenario_classification.csv", index=False)
```

**Problem**: Files are saved in the root directory, which conflicts with the model training code expecting files in `data/processed/`.

**Recommendation**:
```python
# Add output directory configuration
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Update save paths
df_binary.to_csv(f"{OUTPUT_DIR}/binary_balanced.csv", index=False)
df_attack.to_csv(f"{OUTPUT_DIR}/multiclass_balanced.csv", index=False)
df_scenario.to_csv(f"{OUTPUT_DIR}/scenario_balanced.csv", index=False)
```

**Rationale**: The model training code (`model.py`, `enhanced_training.py`) expects these file names and locations.

---

### 3. **ENHANCEMENT: Add Random Seed Configuration**

**Issue**: Random seeds are hardcoded throughout
```python
balancer = SMOTE(random_state=42, k_neighbors=k_neighbors)  # Line 842
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Line 1144
```

**Recommendation**:
```python
# Add at the top of file
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# In class __init__:
def __init__(self, filepath, random_seed=42):
    self.filepath = filepath
    self.random_seed = random_seed
    # ... rest of init

# Use self.random_seed everywhere:
balancer = SMOTE(random_state=self.random_seed, k_neighbors=k_neighbors)
train_test_split(X, y, test_size=0.2, random_state=self.random_seed, stratify=y)
```

**Impact**: Improves reproducibility for the research paper.

---

### 4. **CRITICAL: Add Data Versioning and Metadata**

**Issue**: No metadata tracking for dataset versions

**Recommendation**: Add method to save preprocessing metadata
```python
def save_preprocessing_metadata(self, datasets, output_dir="data/processed"):
    """Save metadata about the preprocessing pipeline for reproducibility"""
    metadata = {
        "preprocessing_date": datetime.now().isoformat(),
        "random_seed": self.random_seed,
        "original_shape": self.df_original.shape,
        "cleaned_shape": self.df_clean.shape,
        "feature_columns": self.feature_columns,
        "attack_mapping": {
            "Cryptojacking": ["cryptojacking"],
            "Reconnaissance": ["aggressive-scan", "os-fingerprinting", "port-scan",
                              "service-detection", "vuln-scan", "os-scan"],
            "DoS": ["icmp-flood", "icmp-fragmentation", "push-ack-flood", "syn-flood",
                   "syn-stealth", "tcp-flood", "udp-flood", "synonymous-ip-flood"],
            "Benign": ["none"]
        },
        "balancing_method": "random_oversample",
        "dataset_shapes": {
            name: df.shape for name, df in datasets.items()
        },
        "class_distributions": {
            "binary": datasets["binary"]["Label"].value_counts().to_dict(),
            "multiclass": datasets.get("attack", {}).get("Attack", pd.Series()).value_counts().to_dict() if "attack" in datasets else {},
            "scenario": datasets["scenario"]["Scenario"].value_counts().to_dict()
        }
    }

    import json
    with open(f"{output_dir}/preprocessing_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved preprocessing metadata to {output_dir}/preprocessing_metadata.json")
    return metadata
```

**Impact**: Essential for reproducibility in academic papers. Reviewers can verify exact preprocessing steps.

---

### 5. **IMPORTANT: Fix Typo in Attack Mapping**

**Issue**: Line 159
```python
"serice-detection": "Reconnaissance",  # TYPO: should be "service-detection"
```

**Recommendation**: Keep both mappings to handle potential data inconsistencies
```python
attack_mapping = {
    "none": "Benign",
    "cryptojacking": "Cryptojacking",
    "aggressive-scan": "Reconnaissance",
    "os-fingerprinting": "Reconnaissance",
    "port-scan": "Reconnaissance",
    "serice-detection": "Reconnaissance",   # Legacy typo - keep for compatibility
    "service-detection": "Reconnaissance",  # Correct spelling
    "vuln-scan": "Reconnaissance",
    # ... rest
}
```

**Impact**: Prevents data loss if both spellings exist in the dataset.

---

### 6. **ENHANCEMENT: Add Non-IID Data Splitting for Federated Learning**

**Issue**: No support for creating non-IID federated client datasets

**Recommendation**: Add new method for federated data splitting
```python
def create_federated_splits(self, num_clients=5, split_strategy="iid", alpha=0.5):
    """
    Create non-IID data splits for federated learning simulation

    Args:
        num_clients: Number of federated clients
        split_strategy: "iid" (balanced) or "non-iid" (imbalanced by attack type)
        alpha: Dirichlet distribution parameter for non-IID (lower = more heterogeneous)

    Returns:
        List of client datasets
    """
    print(f"\nCreating federated data splits: {split_strategy.upper()}")

    X = self.df_clean[self.feature_columns]
    y = self.df_clean["Label"]

    if split_strategy == "iid":
        # Balanced IID split
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=num_clients, shuffle=True, random_state=self.random_seed)

        client_data = []
        for train_idx, _ in kf.split(X):
            client_df = pd.concat([
                X.iloc[train_idx],
                y.iloc[train_idx]
            ], axis=1)
            client_data.append(client_df)

    else:  # non-iid
        # Dirichlet distribution for heterogeneous splits
        from numpy.random import dirichlet

        client_data = [pd.DataFrame() for _ in range(num_clients)]

        # Split each attack type independently with Dirichlet distribution
        for attack_type in self.df_clean["Attack"].unique():
            mask = self.df_clean["Attack"] == attack_type
            attack_data = pd.concat([
                X[mask],
                self.df_clean.loc[mask, ["Label", "Attack", "Scenario"]]
            ], axis=1)

            # Generate proportions using Dirichlet
            proportions = dirichlet([alpha] * num_clients)

            # Assign data to clients based on proportions
            indices = np.arange(len(attack_data))
            np.random.shuffle(indices)

            start_idx = 0
            for client_id, prop in enumerate(proportions):
                end_idx = start_idx + int(prop * len(attack_data))
                client_indices = indices[start_idx:end_idx]
                client_data[client_id] = pd.concat([
                    client_data[client_id],
                    attack_data.iloc[client_indices]
                ])
                start_idx = end_idx

    # Save client datasets
    output_dir = "data/federated_clients"
    os.makedirs(output_dir, exist_ok=True)

    for client_id, client_df in enumerate(client_data):
        filepath = f"{output_dir}/client_{client_id}_{split_strategy}.csv"
        client_df.to_csv(filepath, index=False)
        print(f"✓ Client {client_id}: {len(client_df):,} samples")
        print(f"  Attack distribution: {client_df['Attack'].value_counts().to_dict()}")

    return client_data
```

**Impact**: Enables realistic federated learning experiments for the paper with heterogeneous client data.

---

### 7. **ENHANCEMENT: Add Statistical Analysis for Paper**

**Recommendation**: Add method to generate statistics table for paper
```python
def generate_statistics_table_for_paper(self, datasets):
    """Generate LaTeX table with dataset statistics for paper"""

    stats = []

    # Original dataset statistics
    stats.append({
        "Dataset": "Original",
        "Samples": len(self.df_original),
        "Features": self.df_original.shape[1] - 4,
        "Benign": len(self.df_clean[self.df_clean["Label"] == 0]),
        "Attack": len(self.df_clean[self.df_clean["Label"] == 1]),
        "Cryptojacking": len(self.df_clean[self.df_clean["Attack"] == "Cryptojacking"]),
        "DoS": len(self.df_clean[self.df_clean["Attack"] == "DoS"]),
        "Reconnaissance": len(self.df_clean[self.df_clean["Attack"] == "Reconnaissance"])
    })

    # Cleaned dataset statistics
    stats.append({
        "Dataset": "Cleaned",
        "Samples": len(self.df_clean),
        "Features": len(self.feature_columns),
        "Benign": len(self.df_clean[self.df_clean["Label"] == 0]),
        "Attack": len(self.df_clean[self.df_clean["Label"] == 1]),
        "Cryptojacking": len(self.df_clean[self.df_clean["Attack"] == "Cryptojacking"]),
        "DoS": len(self.df_clean[self.df_clean["Attack"] == "DoS"]),
        "Reconnaissance": len(self.df_clean[self.df_clean["Attack"] == "Reconnaissance"])
    })

    # Balanced datasets
    for name, df in datasets.items():
        if name == "binary":
            label_counts = df["Label"].value_counts()
            stats.append({
                "Dataset": "Binary (Balanced)",
                "Samples": len(df),
                "Features": len(self.feature_columns),
                "Benign": label_counts.get(0, 0),
                "Attack": label_counts.get(1, 0),
                "Cryptojacking": "-",
                "DoS": "-",
                "Reconnaissance": "-"
            })
        elif name == "attack":
            attack_counts = df["Attack"].value_counts()
            stats.append({
                "Dataset": "Multiclass (Balanced)",
                "Samples": len(df),
                "Features": len(self.feature_columns),
                "Benign": "-",
                "Attack": len(df),
                "Cryptojacking": attack_counts.get("Cryptojacking", 0),
                "DoS": attack_counts.get("DoS", 0),
                "Reconnaissance": attack_counts.get("Reconnaissance", 0)
            })

    # Generate LaTeX table
    latex_table = "\\begin{table}[h]\\n"
    latex_table += "\\centering\\n"
    latex_table += "\\caption{Dataset Statistics After Preprocessing}\\n"
    latex_table += "\\label{tab:dataset_statistics}\\n"
    latex_table += "\\begin{tabular}{lrrrrrrr}\\n"
    latex_table += "\\toprule\\n"
    latex_table += "Dataset & Samples & Features & Benign & Attack & Crypto & DoS & Recon \\\\\\n"
    latex_table += "\\midrule\\n"

    for row in stats:
        latex_table += f"{row['Dataset']} & "
        latex_table += f"{row['Samples']:,} & "
        latex_table += f"{row['Features']} & "
        latex_table += f"{row['Benign']:,} & " if row['Benign'] != '-' else "- & "
        latex_table += f"{row['Attack']:,} & " if row['Attack'] != '-' else "- & "
        latex_table += f"{row['Cryptojacking']:,} & " if row['Cryptojacking'] != '-' else "- & "
        latex_table += f"{row['DoS']:,} & " if row['DoS'] != '-' else "- & "
        latex_table += f"{row['Reconnaissance']:,} \\\\\\\\\n" if row['Reconnaissance'] != '-' else "- \\\\\\\\\n"

    latex_table += "\\bottomrule\\n"
    latex_table += "\\end{tabular}\\n"
    latex_table += "\\end{table}\\n"

    # Save to file
    with open("data-results/dataset_statistics_table.tex", "w") as f:
        f.write(latex_table)

    print("✓ Generated LaTeX table: data-results/dataset_statistics_table.tex")

    # Also print to console
    print("\n" + "=" * 80)
    print("DATASET STATISTICS TABLE (for paper)")
    print("=" * 80)
    print(latex_table)

    return stats, latex_table
```

**Impact**: Automatically generates camera-ready LaTeX tables for the paper.

---

### 8. **ENHANCEMENT: Add Progress Bars**

**Recommendation**: Add tqdm progress bars for long operations
```python
# At top of file
from tqdm import tqdm

# In clean_data method (line 224-238):
print("Removing outliers using IQR method...")
numeric_cols = X_clean.select_dtypes(include=[np.number]).columns

outlier_mask = pd.Series([False] * len(X_clean))
for col in tqdm(numeric_cols, desc="Computing outliers"):
    Q1 = X_clean[col].quantile(0.25)
    Q3 = X_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    col_outliers = (X_clean[col] < (Q1 - 1.5 * IQR)) | (X_clean[col] > (Q3 + 1.5 * IQR))
    outlier_mask |= col_outliers
```

**Impact**: Better user experience for long-running preprocessing.

---

### 9. **CRITICAL: Add Error Handling for Missing Data**

**Issue**: No validation that required columns exist

**Recommendation**:
```python
def validate_data(self):
    """Validate that required columns exist"""
    required_columns = ["Label", "Attack", "Scenario", "State"]
    missing_columns = [col for col in required_columns if col not in self.df_original.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print(f"✓ All required columns present: {required_columns}")

def clean_data(self):
    # Add validation at start
    self.validate_data()

    # ... rest of clean_data method
```

**Impact**: Prevents cryptic errors when loading incorrect data files.

---

### 10. **ENHANCEMENT: Add Temporal Sequence Generation**

**Issue**: No support for creating temporal sequences required by TCN

**Recommendation**: Add method to create sliding window sequences
```python
def create_temporal_sequences(self, df, sequence_length=30, stride=1):
    """
    Create temporal sequences for TCN input

    Args:
        df: DataFrame with features and labels
        sequence_length: Number of time steps (default: 30)
        stride: Step size for sliding window (default: 1)

    Returns:
        X_sequences: [n_sequences, sequence_length, n_features]
        y_sequences: [n_sequences]
    """
    print(f"Creating temporal sequences (length={sequence_length}, stride={stride})...")

    X = df[self.feature_columns].values

    # Determine target column
    if "Label" in df.columns:
        y = df["Label"].values
    elif "Attack" in df.columns:
        # Encode attack types
        le = LabelEncoder()
        y = le.fit_transform(df["Attack"].values)
    elif "Scenario" in df.columns:
        le = LabelEncoder()
        y = le.fit_transform(df["Scenario"].values)
    else:
        raise ValueError("No valid target column found")

    sequences = []
    labels = []

    for i in range(0, len(X) - sequence_length + 1, stride):
        sequence = X[i:i+sequence_length]
        label = y[i+sequence_length-1]  # Label of last time step
        sequences.append(sequence)
        labels.append(label)

    X_sequences = np.array(sequences)
    y_sequences = np.array(labels)

    print(f"✓ Created {len(sequences):,} sequences")
    print(f"  Shape: {X_sequences.shape}")

    return X_sequences, y_sequences

def save_sequences_for_pytorch(self, datasets, sequence_length=30):
    """Save temporal sequences in PyTorch-compatible format"""
    import torch

    output_dir = "data/sequences"
    os.makedirs(output_dir, exist_ok=True)

    for name, df in datasets.items():
        X_seq, y_seq = self.create_temporal_sequences(df, sequence_length)

        # Save as PyTorch tensors
        torch.save({
            'X': torch.FloatTensor(X_seq),
            'y': torch.LongTensor(y_seq),
            'feature_names': self.feature_columns,
            'sequence_length': sequence_length
        }, f"{output_dir}/{name}_sequences.pt")

        print(f"✓ Saved {name} sequences to {output_dir}/{name}_sequences.pt")
```

**Impact**: Makes data directly compatible with the TCN model without additional preprocessing.

---

### 11. **DOCUMENTATION: Add Comprehensive Docstrings**

**Issue**: Some methods lack detailed docstrings

**Recommendation**: Add Google-style docstrings to all methods
```python
def clean_data(self):
    """
    Clean and preprocess EVSE dataset.

    Performs the following operations:
    1. Filters invalid records (missing labels, attacks)
    2. Encodes target variables (binary, multiclass)
    3. Maps attack variants to 4 categories
    4. Removes problematic columns (constant, high missing, duplicates)
    5. Handles missing values (median for numeric, mode for categorical)
    6. Removes outliers using IQR method (1.5 * IQR threshold)

    Returns:
        pd.DataFrame: Cleaned dataset with shape [n_samples, n_features + 4_targets]

    Raises:
        ValueError: If no valid records remain after filtering

    Example:
        >>> processor = EVSEDataProcessor("data.csv")
        >>> processor.load_data()
        >>> df_clean = processor.clean_data()
        >>> print(df_clean.shape)
        (150000, 45)
    """
    # ... existing implementation
```

---

## Priority Implementation Order

### Phase 1: Critical Fixes (Do immediately)
1. ✅ Fix output file paths to `data/processed/` with correct names
2. ✅ Add preprocessing metadata saving
3. ✅ Add data validation method
4. ✅ Fix directory naming consistency (`DATA_RESULTS_DIR`)

### Phase 2: Federated Learning Support (This week)
5. ✅ Add non-IID data splitting for federated clients
6. ✅ Add temporal sequence generation for TCN
7. ✅ Add random seed configuration

### Phase 3: Paper Enhancements (Next week)
8. ✅ Add LaTeX statistics table generation
9. ✅ Add progress bars with tqdm
10. ✅ Add comprehensive docstrings

---

## Code Quality Improvements

### Testing Recommendations
```python
# Add at end of file for unit testing
def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline"""
    processor = EVSEDataProcessor("./data/Host Events/EVSE-B-HPC-Kernel-Events-Combined.csv")

    # Test data loading
    df = processor.load_data()
    assert df is not None, "Failed to load data"
    assert len(df) > 0, "Loaded empty dataset"

    # Test data cleaning
    df_clean = processor.clean_data()
    assert len(df_clean) > 0, "Cleaning removed all data"
    assert df_clean.isnull().sum().sum() == 0, "Missing values remain"

    # Test dataset creation
    datasets = processor.create_ml_datasets()
    assert "binary" in datasets, "Binary dataset not created"
    assert "scenario" in datasets, "Scenario dataset not created"

    print("✓ All tests passed!")

    return True

# Uncomment to run tests:
# if __name__ == "__main__":
#     test_preprocessing_pipeline()
```

---

## Integration with Enhanced Framework

### Alignment with `enhanced_training.py`

The enhanced training code expects files in this format:
```
data/processed/
├── binary_balanced.csv      # Binary classification
├── multiclass_balanced.csv  # Multiclass (attack types)
└── scenario_balanced.csv    # Scenario classification
```

**Required Changes in data.py**:
```python
# Update line 877-921 in create_ml_datasets():
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save with correct names
df_binary.to_csv(f"{OUTPUT_DIR}/binary_balanced.csv", index=False)
df_attack.to_csv(f"{OUTPUT_DIR}/multiclass_balanced.csv", index=False)
df_scenario.to_csv(f"{OUTPUT_DIR}/scenario_balanced.csv", index=False)
```

---

## Summary of Required Changes

### Must-Do (Before Running Experiments)
- [x] Fix output directory paths to `data/processed/`
- [x] Rename output files: `binary_balanced.csv`, `multiclass_balanced.csv`, `scenario_balanced.csv`
- [x] Add preprocessing metadata JSON export
- [x] Add data validation method

### Should-Do (For Paper Quality)
- [ ] Add LaTeX table generation for statistics
- [ ] Add non-IID federated data splitting
- [ ] Add temporal sequence generation
- [ ] Add progress bars for long operations

### Nice-to-Have (For Code Quality)
- [ ] Add comprehensive docstrings
- [ ] Add unit tests
- [ ] Add random seed configuration class
- [ ] Add logging instead of print statements

---

## Estimated Impact

| Improvement | Lines of Code | Time to Implement | Impact on Paper |
|-------------|---------------|-------------------|-----------------|
| Fix file paths | 5 lines | 5 minutes | Critical - enables training |
| Add metadata saving | 30 lines | 20 minutes | High - reproducibility |
| Add federated splits | 80 lines | 45 minutes | High - realistic FL experiments |
| Add LaTeX tables | 60 lines | 30 minutes | Medium - camera-ready tables |
| Add sequence generation | 50 lines | 25 minutes | Medium - direct TCN compat |
| Add docstrings | 100 lines | 60 minutes | Low - code quality |

**Total Estimated Time**: 3-4 hours for all must-do and should-do items

---

## Conclusion

The `data.py` file is production-quality code that needs minor adjustments for optimal integration with the federated learning framework and reproducible research. The critical path is:

1. **Fix output paths** (5 min) ← Do this first!
2. **Add metadata** (20 min) ← Essential for paper
3. **Add federated splits** (45 min) ← Enables non-IID experiments
4. **Add LaTeX tables** (30 min) ← Makes results section easier

These 4 changes will significantly strengthen the paper's experimental validation and reproducibility.
