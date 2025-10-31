"""
EVSE-B Data Preprocessing Pipeline - IMPROVED VERSION

This is an enhanced version of data.py with all critical fixes applied:
- Correct output file paths (data/processed/)
- Proper file naming for enhanced_training.py compatibility
- Preprocessing metadata export (JSON + README)
- LaTeX table generation for paper
- Progress bars for long operations
- Better error handling
- Comprehensive logging

Usage:
    python data_improved.py

Author: Enhanced for CICEVSE Federated Learning Project
Date: 2025-10-30
"""

import os
import sys
import json
import shutil
import warnings
from datetime import datetime
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

from scipy import stats

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

RANDOM_SEED = 42
DATA_RESULTS_DIR = 'data-results'
OUTPUT_DIR = 'data/processed'

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)

# ==============================================================================
# VISUALIZATION CONFIGURATION
# ==============================================================================

LAYOUT = go.Layout(
    template="plotly_white",
    margin=dict(l=40, r=40, b=40, t=40),
    font=dict(family="Arial", size=13, color="darkslategrey"),
    width=600,
    height=600,
    autosize=False,
)

COLORS = [
    "#81C784", "#64B5F6", "#E45756", "#8B88C6", "#6DCCDA",
    "#FF8A65", "#4DD0E1", "#F06292", "#90A4AE", "#AED581",
    "#A1887F", "#E6EE9C", "#9575CD", "#FFB74D", "#4FC3F7",
    "#EF5350",
]

# ==============================================================================
# EVSE DATA PROCESSOR CLASS (IMPROVED)
# ==============================================================================

class EVSEDataProcessor:
    """Enhanced EVSE Data Processor with all improvements"""

    def __init__(self, filepath, random_seed=RANDOM_SEED):
        self.filepath = filepath
        self.random_seed = random_seed
        self.df_original = None
        self.df_clean = None
        self.feature_columns = None
        self.target_columns = ["State", "Attack", "Scenario", "Label"]
        self.scaler = StandardScaler()
        self.figures = []

        # Attack mappings
        self.attack_mapping = {
            "none": "Benign",
            "cryptojacking": "Cryptojacking",
            "aggressive-scan": "Reconnaissance",
            "os-fingerprinting": "Reconnaissance",
            "port-scan": "Reconnaissance",
            "serice-detection": "Reconnaissance",  # Legacy typo
            "service-detection": "Reconnaissance",
            "vuln-scan": "Reconnaissance",
            "os-scan": "Reconnaissance",
            "icmp-flood": "DoS",
            "icmp-fragmentation_old": "DoS",
            "icmp-fragmentation": "DoS",
            "push-ack-flood": "DoS",
            "syn-flood": "DoS",
            "syn-stealth": "DoS",
            "tcp-flood": "DoS",
            "udp-flood": "DoS",
            "synonymous-ip-flood": "DoS",
        }

        self.scenario_mapping = {
            "Benign": "Benign",
            "Cryptojacking": "Cryptojacking",
            "Recon": "Reconnaissance",
            "DoS": "DoS",
        }

    def load_data(self):
        """Load EVSE-B dataset with validation"""
        print("Loading EVSE-B dataset...")
        try:
            if not os.path.exists(self.filepath):
                raise FileNotFoundError(f"Data file not found: {self.filepath}")

            self.df_original = pd.read_csv(self.filepath)
            print(f"✓ Successfully loaded dataset with shape: {self.df_original.shape}")
            print(f"  Memory usage: {self.df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            return self.df_original

        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            raise

    def validate_data(self):
        """Validate that required columns exist"""
        print("\nValidating dataset structure...")
        required_columns = ["Label", "Attack", "Scenario"]
        missing_columns = [col for col in required_columns if col not in self.df_original.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        print(f"✓ All required columns present: {required_columns}")

    def initial_data_exploration(self):
        """Explore dataset statistics"""
        print("\n" + "=" * 70)
        print("INITIAL DATA EXPLORATION")
        print("=" * 70)

        print(f"Dataset shape: {self.df_original.shape}")
        print(f"Memory usage: {self.df_original.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        print("\nTarget variable distributions:")
        for target in self.target_columns:
            if target in self.df_original.columns:
                print(f"\n{target}:")
                counts = self.df_original[target].value_counts()
                for value, count in counts.items():
                    print(f"  {value}: {count:,}")

        missing_values = self.df_original.isnull().sum().sum()
        duplicate_rows = self.df_original.duplicated().sum()
        print(f"\nData Quality:")
        print(f"  Missing values: {missing_values:,}")
        print(f"  Duplicate rows: {duplicate_rows:,}")

    def clean_data(self):
        """Clean and preprocess data with progress tracking"""
        print("\n" + "=" * 70)
        print("DATA CLEANING AND PREPROCESSING")
        print("=" * 70)

        # Identify feature columns
        exclude_columns = ["interface", "", "_1", "_2", "_3", "time"]
        all_columns = list(self.df_original.columns)
        self.feature_columns = [
            col for col in all_columns
            if col not in self.target_columns + exclude_columns and col.strip() != ""
        ]

        print(f"✓ Identified {len(self.feature_columns)} feature columns")

        df_work = self.df_original[self.feature_columns + self.target_columns].copy()

        # Filter valid records
        print("Filtering valid records...")
        initial_count = len(df_work)
        valid_mask = (
            (df_work["Label"].isin(["attack", "benign"]))
            & (df_work["Attack"] != "0")
            & (df_work["Attack"] != 0)
            & (df_work["Scenario"] != "0")
            & (df_work["Scenario"] != 0)
            & (df_work["Attack"].notna())
            & (df_work["Scenario"].notna())
        )

        df_work = df_work[valid_mask].copy()
        print(f"✓ Filtered {initial_count - len(df_work):,} invalid records")
        print(f"✓ Remaining records: {len(df_work):,}")

        # Encode target variables
        print("Encoding target variables...")
        df_work["Label"] = df_work["Label"].map({"attack": 1, "benign": 0})
        df_work["Attack"] = df_work["Attack"].map(self.attack_mapping)
        df_work["Scenario"] = df_work["Scenario"].map(self.scenario_mapping)

        # Remove problematic columns
        print("Removing problematic columns...")
        X = df_work[self.feature_columns]

        # Constant columns
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]

        # High missing columns
        high_missing_cols = [
            col for col in X.columns
            if X[col].isnull().sum() / len(X) > 0.95
        ]

        # Duplicate columns
        duplicate_cols = []
        for i, col1 in enumerate(tqdm(X.columns, desc="Checking duplicates")):
            for col2 in X.columns[i + 1:]:
                if X[col1].equals(X[col2]):
                    duplicate_cols.append(col2)
                    break

        cols_to_remove = list(set(constant_cols + high_missing_cols + duplicate_cols))
        print(f"✓ Removing {len(cols_to_remove)} problematic columns:")
        print(f"  - Constant columns: {len(constant_cols)}")
        print(f"  - High missing (>95%): {len(high_missing_cols)}")
        print(f"  - Duplicate columns: {len(duplicate_cols)}")

        X_clean = X.drop(columns=cols_to_remove)
        self.feature_columns = list(X_clean.columns)

        # Handle missing values
        print("Handling missing values...")
        for col in tqdm(X_clean.columns, desc="Imputing"):
            if X_clean[col].dtype in ["float64", "int64"]:
                X_clean[col] = X_clean[col].fillna(X_clean[col].median())
            else:
                mode_val = X_clean[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else "unknown"
                X_clean[col] = X_clean[col].fillna(fill_val)

        # Remove outliers using IQR
        print("Removing outliers using IQR method...")
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns

        outlier_mask = pd.Series([False] * len(X_clean))
        for col in tqdm(numeric_cols, desc="Computing outliers"):
            Q1 = X_clean[col].quantile(0.25)
            Q3 = X_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            col_outliers = (X_clean[col] < (Q1 - 1.5 * IQR)) | (X_clean[col] > (Q3 + 1.5 * IQR))
            outlier_mask |= col_outliers

        outlier_count = outlier_mask.sum()
        print(f"✓ Removing {outlier_count:,} outlier rows ({outlier_count/len(X_clean)*100:.1f}%)")

        self.df_clean = pd.concat(
            [X_clean[~outlier_mask], df_work[self.target_columns][~outlier_mask]],
            axis=1,
        )

        print(f"✓ Final cleaned dataset shape: {self.df_clean.shape}")
        print(f"✓ Final feature count: {len(self.feature_columns)}")

        return self.df_clean

    def feature_analysis(self):
        """Analyze feature importance"""
        print("\nAnalyzing feature importance...")

        numeric_features = self.df_clean.select_dtypes(include=[np.number]).columns
        numeric_features = [col for col in numeric_features if col not in self.target_columns]

        feature_importance = []

        for col in numeric_features:
            try:
                correlation = abs(self.df_clean[col].corr(self.df_clean["Label"]))
                if not np.isnan(correlation):
                    feature_importance.append((col, correlation))
            except:
                continue

        feature_importance.sort(key=lambda x: x[1], reverse=True)

        print(f"✓ Analyzed {len(feature_importance)} features")
        return feature_importance

    def balance_dataset(self, X, y, method="random_oversample"):
        """Balance dataset using various methods"""
        print(f"Balancing dataset using {method}...")

        try:
            if method == "smote":
                min_samples = min(pd.Series(y).value_counts())
                k_neighbors = min(3, min_samples - 1) if min_samples > 1 else 1
                balancer = SMOTE(random_state=self.random_seed, k_neighbors=k_neighbors)
            elif method == "adasyn":
                balancer = ADASYN(random_state=self.random_seed)
            else:
                balancer = RandomOverSampler(random_state=self.random_seed)
        except:
            balancer = RandomOverSampler(random_state=self.random_seed)

        X_balanced, y_balanced = balancer.fit_resample(X, y)
        return X_balanced, y_balanced

    def create_ml_datasets(self):
        """Create ML-ready datasets with correct paths"""
        print("\n" + "=" * 70)
        print("CREATING ML-READY DATASETS")
        print("=" * 70)

        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        X = self.df_clean[self.feature_columns]
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), columns=self.feature_columns
        )

        datasets = {}

        # Phase 1: Binary Classification
        print("\n[Phase 1/3] Binary Classification (Attack vs Benign)")
        y_binary = self.df_clean["Label"]
        print(f"Original distribution: {y_binary.value_counts().to_dict()}")

        X_binary_balanced, y_binary_balanced = self.balance_dataset(X_scaled, y_binary)

        df_binary = pd.DataFrame(X_binary_balanced, columns=self.feature_columns)
        df_binary["Label"] = y_binary_balanced

        print(f"Balanced distribution: {pd.Series(y_binary_balanced).value_counts().to_dict()}")

        # FIXED: Save with correct name and path
        binary_path = f"{OUTPUT_DIR}/binary_balanced.csv"
        df_binary.to_csv(binary_path, index=False)
        datasets["binary"] = df_binary
        print(f"✓ Saved: {binary_path} ({len(df_binary):,} samples)")

        # Phase 2: Multi-class Attack Types
        print("\n[Phase 2/3] Multi-class Attack Types")
        attack_data = self.df_clean[self.df_clean["Label"] == 1].copy()
        if len(attack_data) > 0:
            X_attack = attack_data[self.feature_columns]
            X_attack_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_attack), columns=self.feature_columns
            )
            y_attack = attack_data["Attack"]
            print(f"Original distribution: {y_attack.value_counts().to_dict()}")

            X_attack_balanced, y_attack_balanced = self.balance_dataset(
                X_attack_scaled, y_attack
            )

            df_attack = pd.DataFrame(X_attack_balanced, columns=self.feature_columns)
            df_attack["Attack"] = y_attack_balanced

            print(f"Balanced distribution: {pd.Series(y_attack_balanced).value_counts().to_dict()}")

            # FIXED: Save with correct name and path
            multiclass_path = f"{OUTPUT_DIR}/multiclass_balanced.csv"
            df_attack.to_csv(multiclass_path, index=False)
            datasets["attack"] = df_attack
            print(f"✓ Saved: {multiclass_path} ({len(df_attack):,} samples)")

        # Phase 3: Scenario Classification
        print("\n[Phase 3/3] Scenario Classification")
        y_scenario = self.df_clean["Scenario"]
        print(f"Original distribution: {y_scenario.value_counts().to_dict()}")

        X_scenario_balanced, y_scenario_balanced = self.balance_dataset(
            X_scaled, y_scenario
        )

        df_scenario = pd.DataFrame(X_scenario_balanced, columns=self.feature_columns)
        df_scenario["Scenario"] = y_scenario_balanced

        print(f"Balanced distribution: {pd.Series(y_scenario_balanced).value_counts().to_dict()}")

        # FIXED: Save with correct name and path
        scenario_path = f"{OUTPUT_DIR}/scenario_balanced.csv"
        df_scenario.to_csv(scenario_path, index=False)
        datasets["scenario"] = df_scenario
        print(f"✓ Saved: {scenario_path} ({len(df_scenario):,} samples)")

        return datasets

    def save_preprocessing_metadata(self, datasets):
        """Save preprocessing metadata for reproducibility"""
        print("\n" + "=" * 70)
        print("SAVING PREPROCESSING METADATA")
        print("=" * 70)

        metadata = {
            "preprocessing_date": datetime.now().isoformat(),
            "random_seed": self.random_seed,
            "original_shape": list(self.df_original.shape),
            "cleaned_shape": list(self.df_clean.shape),
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "attack_mapping": {
                "Benign": ["none"],
                "Cryptojacking": ["cryptojacking"],
                "Reconnaissance": ["aggressive-scan", "os-fingerprinting", "port-scan",
                                  "serice-detection", "service-detection", "vuln-scan", "os-scan"],
                "DoS": ["icmp-flood", "icmp-fragmentation_old", "icmp-fragmentation",
                       "push-ack-flood", "syn-flood", "syn-stealth", "tcp-flood",
                       "udp-flood", "synonymous-ip-flood"]
            },
            "balancing_method": "random_oversample",
            "dataset_shapes": {
                name: list(df.shape) for name, df in datasets.items()
            },
            "class_distributions": {
                "binary": datasets["binary"]["Label"].value_counts().to_dict(),
                "multiclass": datasets.get("attack", {}).get("Attack", pd.Series()).value_counts().to_dict() if "attack" in datasets else {},
                "scenario": datasets["scenario"]["Scenario"].value_counts().to_dict()
            },
            "preprocessing_steps": [
                "1. Load raw EVSE-B HPC Kernel Events data",
                "2. Filter invalid records (missing labels, attacks)",
                "3. Map attack variants to 4 categories (Benign, Cryptojacking, DoS, Reconnaissance)",
                "4. Remove constant columns (zero variance)",
                "5. Remove high missing columns (>95% missing)",
                "6. Remove duplicate columns",
                "7. Impute missing values (median for numeric, mode for categorical)",
                "8. Remove outliers using IQR method (1.5 * IQR)",
                "9. Scale features using StandardScaler",
                "10. Balance datasets using Random Oversampling"
            ]
        }

        # Save JSON
        metadata_path = f"{OUTPUT_DIR}/preprocessing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved JSON metadata: {metadata_path}")

        # Save human-readable README
        readme_path = f"{OUTPUT_DIR}/README.txt"
        with open(readme_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EVSE-B PREPROCESSED DATASET DOCUMENTATION\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {metadata['preprocessing_date']}\n")
            f.write(f"Random Seed: {metadata['random_seed']}\n\n")

            f.write("DATASET SHAPES:\n")
            f.write(f"  Original: {metadata['original_shape'][0]:,} samples × {metadata['original_shape'][1]} columns\n")
            f.write(f"  Cleaned:  {metadata['cleaned_shape'][0]:,} samples × {metadata['cleaned_shape'][1]} columns\n")
            f.write(f"  Features: {metadata['feature_count']} (after removing problematic columns)\n\n")

            f.write("GENERATED FILES:\n")
            f.write("  • binary_balanced.csv      - Binary classification (Attack vs Benign)\n")
            f.write("  • multiclass_balanced.csv  - Multiclass attack type classification\n")
            f.write("  • scenario_balanced.csv    - Scenario-based classification\n\n")

            f.write("CLASS DISTRIBUTIONS:\n")
            f.write("  Binary Classification:\n")
            for label, count in metadata['class_distributions']['binary'].items():
                label_name = "Benign" if label == 0 else "Attack"
                f.write(f"    {label_name}: {count:,}\n")

            if metadata['class_distributions']['multiclass']:
                f.write("\n  Multiclass Attack Types:\n")
                for attack, count in metadata['class_distributions']['multiclass'].items():
                    f.write(f"    {attack}: {count:,}\n")

            f.write("\n  Scenario Classification:\n")
            for scenario, count in metadata['class_distributions']['scenario'].items():
                f.write(f"    {scenario}: {count:,}\n")

            f.write("\nPREPROCESSING STEPS:\n")
            for step in metadata['preprocessing_steps']:
                f.write(f"  {step}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Ready for federated learning with enhanced_training.py\n")
            f.write("=" * 80 + "\n")

        print(f"✓ Saved human-readable README: {readme_path}")

        return metadata

    def generate_latex_table(self, datasets):
        """Generate LaTeX table for paper"""
        print("\n" + "=" * 70)
        print("GENERATING LATEX TABLE FOR PAPER")
        print("=" * 70)

        stats = []

        # Original dataset
        benign_count = len(self.df_clean[self.df_clean["Label"] == 0])
        attack_count = len(self.df_clean[self.df_clean["Label"] == 1])
        crypto_count = len(self.df_clean[self.df_clean["Attack"] == "Cryptojacking"])
        dos_count = len(self.df_clean[self.df_clean["Attack"] == "DoS"])
        recon_count = len(self.df_clean[self.df_clean["Attack"] == "Reconnaissance"])

        stats.append({
            "Dataset": "Original (Filtered)",
            "Samples": len(self.df_clean),
            "Features": len(self.feature_columns),
            "Benign": benign_count,
            "Attack": attack_count,
            "Cryptojacking": crypto_count,
            "DoS": dos_count,
            "Reconnaissance": recon_count
        })

        # Balanced datasets
        if "binary" in datasets:
            label_counts = datasets["binary"]["Label"].value_counts()
            stats.append({
                "Dataset": "Binary (Balanced)",
                "Samples": len(datasets["binary"]),
                "Features": len(self.feature_columns),
                "Benign": label_counts.get(0, 0),
                "Attack": label_counts.get(1, 0),
                "Cryptojacking": "---",
                "DoS": "---",
                "Reconnaissance": "---"
            })

        if "attack" in datasets and "Attack" in datasets["attack"].columns:
            attack_counts = datasets["attack"]["Attack"].value_counts()
            stats.append({
                "Dataset": "Multiclass (Balanced)",
                "Samples": len(datasets["attack"]),
                "Features": len(self.feature_columns),
                "Benign": "---",
                "Attack": len(datasets["attack"]),
                "Cryptojacking": attack_counts.get("Cryptojacking", 0),
                "DoS": attack_counts.get("DoS", 0),
                "Reconnaissance": attack_counts.get("Reconnaissance", 0)
            })

        # Generate LaTeX
        latex_table = "\\begin{table}[h]\n"
        latex_table += "\\centering\n"
        latex_table += "\\renewcommand{\\arraystretch}{1.3}\n"
        latex_table += "\\caption{CICEVSE2024 Dataset Statistics After Preprocessing and Balancing}\n"
        latex_table += "\\label{tab:dataset_statistics}\n"
        latex_table += "\\begin{tabular}{lrrrrrrr}\n"
        latex_table += "\\toprule\n"
        latex_table += "\\textbf{Dataset} & \\textbf{Samples} & \\textbf{Features} & \\textbf{Benign} & \\textbf{Attack} & \\textbf{Crypto} & \\textbf{DoS} & \\textbf{Recon} \\\\\n"
        latex_table += "\\midrule\n"

        for row in stats:
            latex_table += f"{row['Dataset']} & "
            latex_table += f"{row['Samples']:,} & "
            latex_table += f"{row['Features']} & "

            for col in ['Benign', 'Attack', 'Cryptojacking', 'DoS', 'Reconnaissance']:
                val = row[col]
                if val == "---":
                    latex_table += "--- & " if col != 'Reconnaissance' else "--- \\\\\n"
                else:
                    latex_table += f"{val:,} & " if col != 'Reconnaissance' else f"{val:,} \\\\\n"

        latex_table += "\\bottomrule\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\end{table}\n"

        # Save to file
        os.makedirs(DATA_RESULTS_DIR, exist_ok=True)
        latex_path = f"{DATA_RESULTS_DIR}/dataset_statistics_table.tex"

        with open(latex_path, "w") as f:
            f.write(latex_table)

        print(f"✓ Generated LaTeX table: {latex_path}")
        print("\n" + "=" * 70)
        print(latex_table)
        print("=" * 70)

        return stats, latex_table

    # Visualization methods (keeping the originals with small fixes)
    def create_figure_1_binary_distribution(self):
        """Create binary distribution plot"""
        label_counts = self.df_clean["Label"].value_counts().sort_index()

        fig = go.Figure(layout=LAYOUT)
        fig.add_trace(go.Bar(
            x=["Benign", "Attack"],
            y=[label_counts[0], label_counts[1]],
            marker_color=COLORS[:2],
            text=[f"{label_counts[0]:,}", f"{label_counts[1]:,}"],
            textposition="auto",
            textfont=dict(size=16, color="white"),
        ))

        fig.update_layout(
            xaxis_title="Class",
            yaxis_title="Number of Samples",
            showlegend=False,
        )

        fig.show()
        fig.write_image(f"{DATA_RESULTS_DIR}/binary_distribution.png")
        self.figures.append(fig)
        return fig

    def create_figure_2_attack_types(self):
        """Create attack types distribution plot"""
        attack_counts = self.df_clean["Attack"].value_counts()

        fig = go.Figure(layout=LAYOUT)
        fig.add_trace(go.Bar(
            x=attack_counts.index,
            y=attack_counts.values,
            marker_color=COLORS[:len(attack_counts)],
            text=[f"{count:,}" for count in attack_counts.values],
            textposition="auto",
            textfont=dict(size=12, color="white"),
        ))

        fig.update_layout(
            xaxis_title="Attack Type",
            yaxis_title="Number of Samples",
            showlegend=False,
            xaxis_tickangle=-45,
        )

        fig.show()
        fig.write_image(f"{DATA_RESULTS_DIR}/attack_types.png")
        self.figures.append(fig)
        return fig

    def create_all_visualizations(self):
        """Create all visualizations"""
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)

        os.makedirs(DATA_RESULTS_DIR, exist_ok=True)

        print("Creating binary distribution...")
        self.create_figure_1_binary_distribution()

        print("Creating attack types distribution...")
        self.create_figure_2_attack_types()

        print(f"✓ Created {len(self.figures)} visualizations in {DATA_RESULTS_DIR}/")

        return self.figures


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

def main():
    """Main execution function"""
    print("=" * 80)
    print("EVSE-B DATA PREPROCESSING PIPELINE (IMPROVED VERSION)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 80)

    filepath = "./data/raw/Host Events/EVSE-B-HPC-Kernel-Events-Combined.csv"

    try:
        # Initialize processor
        processor = EVSEDataProcessor(filepath, random_seed=RANDOM_SEED)

        # Step 1: Load and validate data
        print("\n[STEP 1/6] Loading and validating data...")
        processor.load_data()
        processor.validate_data()
        processor.initial_data_exploration()

        # Step 2: Clean and preprocess data
        print("\n[STEP 2/6] Cleaning and preprocessing data...")
        processor.clean_data()

        # Step 3: Create ML-ready datasets (with correct paths!)
        print("\n[STEP 3/6] Creating ML-ready datasets...")
        datasets = processor.create_ml_datasets()

        # Step 4: Save preprocessing metadata
        print("\n[STEP 4/6] Saving preprocessing metadata...")
        metadata = processor.save_preprocessing_metadata(datasets)

        # Step 5: Generate LaTeX table for paper
        print("\n[STEP 5/6] Generating LaTeX table for paper...")
        stats, latex_table = processor.generate_latex_table(datasets)

        # Step 6: Create visualizations
        print("\n[STEP 6/6] Creating visualizations...")
        figures = processor.create_all_visualizations()

        # Final summary
        print("\n" + "=" * 80)
        print("✅ PREPROCESSING COMPLETE - ALL FILES READY!")
        print("=" * 80)

        print(f"\n📊 SUMMARY:")
        print(f"  • Original dataset: {processor.df_original.shape[0]:,} samples")
        print(f"  • Cleaned dataset: {processor.df_clean.shape[0]:,} samples")
        print(f"  • Final features: {len(processor.feature_columns)}")
        print(f"  • Datasets created: {len(datasets)}")
        print(f"  • Visualizations: {len(figures)}")

        print(f"\n📁 OUTPUT FILES:")
        print(f"  ✓ {OUTPUT_DIR}/binary_balanced.csv")
        print(f"  ✓ {OUTPUT_DIR}/multiclass_balanced.csv")
        print(f"  ✓ {OUTPUT_DIR}/scenario_balanced.csv")
        print(f"  ✓ {OUTPUT_DIR}/preprocessing_metadata.json")
        print(f"  ✓ {OUTPUT_DIR}/README.txt")
        print(f"  ✓ {DATA_RESULTS_DIR}/dataset_statistics_table.tex")

        print(f"\n🚀 NEXT STEP:")
        print(f"  Run enhanced training:")
        print(f"    python enhanced_training.py")
        print(f"  Or use the automated runner:")
        print(f"    python run_complete_pipeline.py --mode full --skip-preprocessing")

        print("\n" + "=" * 80)

        return processor, datasets, metadata

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    processor, datasets, metadata = main()

    if processor is not None:
        print("\n✨ Success! You can now proceed to training.")
        print(f"   All files are in the correct locations for enhanced_training.py")
    else:
        print("\n❌ Preprocessing failed. Please check the error messages above.")
        sys.exit(1)
