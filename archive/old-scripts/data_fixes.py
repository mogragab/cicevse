"""
Critical fixes for data.py to integrate with enhanced federated learning framework

Apply these patches to data.py for immediate compatibility with enhanced_training.py
"""

import os
import json
from datetime import datetime

# ==============================================================================
# PATCH 1: Fix output directory configuration
# ==============================================================================

DATA_RESULTS_DIR = 'data-results'
OUTPUT_DIR = 'data/processed'

# Create directories
os.makedirs(DATA_RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# PATCH 2: Add preprocessing metadata saving method
# ==============================================================================

def save_preprocessing_metadata(processor, datasets, output_dir="data/processed"):
    """
    Save metadata about the preprocessing pipeline for reproducibility

    Args:
        processor: EVSEDataProcessor instance
        datasets: Dictionary of processed datasets
        output_dir: Directory to save metadata
    """
    metadata = {
        "preprocessing_date": datetime.now().isoformat(),
        "random_seed": 42,
        "original_shape": list(processor.df_original.shape),
        "cleaned_shape": list(processor.df_clean.shape),
        "feature_count": len(processor.feature_columns),
        "feature_columns": processor.feature_columns,
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
            "multiclass": datasets.get("attack", {}).get("Attack", {}).value_counts().to_dict() if "attack" in datasets and "Attack" in datasets["attack"].columns else {},
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

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = f"{output_dir}/preprocessing_metadata.json"

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved preprocessing metadata to {metadata_path}")

    # Also save human-readable version
    readme_path = f"{output_dir}/README.txt"
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
        f.write("For complete metadata, see preprocessing_metadata.json\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Saved human-readable README to {readme_path}")

    return metadata


# ==============================================================================
# PATCH 3: Generate LaTeX statistics table for paper
# ==============================================================================

def generate_statistics_table_for_paper(processor, datasets, output_dir="data-results"):
    """
    Generate LaTeX table with dataset statistics for paper

    Args:
        processor: EVSEDataProcessor instance
        datasets: Dictionary of processed datasets
        output_dir: Directory to save LaTeX file
    """
    stats = []

    # Original dataset statistics
    benign_count = len(processor.df_clean[processor.df_clean["Label"] == 0])
    attack_count = len(processor.df_clean[processor.df_clean["Label"] == 1])
    crypto_count = len(processor.df_clean[processor.df_clean["Attack"] == "Cryptojacking"])
    dos_count = len(processor.df_clean[processor.df_clean["Attack"] == "DoS"])
    recon_count = len(processor.df_clean[processor.df_clean["Attack"] == "Reconnaissance"])

    stats.append({
        "Dataset": "Original (Filtered)",
        "Samples": len(processor.df_clean),
        "Features": len(processor.feature_columns),
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
            "Features": len(processor.feature_columns),
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
            "Features": len(processor.feature_columns),
            "Benign": "---",
            "Attack": len(datasets["attack"]),
            "Cryptojacking": attack_counts.get("Cryptojacking", 0),
            "DoS": attack_counts.get("DoS", 0),
            "Reconnaissance": attack_counts.get("Reconnaissance", 0)
        })

    if "scenario" in datasets:
        scenario_counts = datasets["scenario"]["Scenario"].value_counts()
        stats.append({
            "Dataset": "Scenario (Balanced)",
            "Samples": len(datasets["scenario"]),
            "Features": len(processor.feature_columns),
            "Benign": scenario_counts.get("Benign", 0),
            "Attack": sum(scenario_counts.get(k, 0) for k in ["Cryptojacking", "DoS", "Reconnaissance"]),
            "Cryptojacking": scenario_counts.get("Cryptojacking", 0),
            "DoS": scenario_counts.get("DoS", 0),
            "Reconnaissance": scenario_counts.get("Reconnaissance", 0)
        })

    # Generate LaTeX table
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
    os.makedirs(output_dir, exist_ok=True)
    latex_path = f"{output_dir}/dataset_statistics_table.tex"

    with open(latex_path, "w") as f:
        f.write(latex_table)

    print(f"✓ Generated LaTeX table: {latex_path}")

    # Print to console
    print("\n" + "=" * 80)
    print("DATASET STATISTICS TABLE (for paper)")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)

    return stats, latex_table


# ==============================================================================
# USAGE INSTRUCTIONS
# ==============================================================================

def apply_fixes_to_data_processing(processor, datasets):
    """
    Apply all critical fixes after running the standard data.py pipeline

    Usage:
        # After running data.py main():
        from data_fixes import apply_fixes_to_data_processing
        apply_fixes_to_data_processing(processor, datasets)

    Args:
        processor: EVSEDataProcessor instance from data.py
        datasets: Dictionary of datasets returned by create_ml_datasets()
    """
    print("\n" + "=" * 80)
    print("APPLYING CRITICAL FIXES FOR FEDERATED LEARNING")
    print("=" * 80)

    # Fix 1: Re-save datasets with correct names and paths
    print("\n[1/3] Fixing output file paths...")

    OUTPUT_DIR = "data/processed"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if "binary" in datasets:
        path = f"{OUTPUT_DIR}/binary_balanced.csv"
        datasets["binary"].to_csv(path, index=False)
        print(f"  ✓ Saved {path} ({len(datasets['binary']):,} samples)")

    if "attack" in datasets:
        path = f"{OUTPUT_DIR}/multiclass_balanced.csv"
        datasets["attack"].to_csv(path, index=False)
        print(f"  ✓ Saved {path} ({len(datasets['attack']):,} samples)")

    if "scenario" in datasets:
        path = f"{OUTPUT_DIR}/scenario_balanced.csv"
        datasets["scenario"].to_csv(path, index=False)
        print(f"  ✓ Saved {path} ({len(datasets['scenario']):,} samples)")

    # Fix 2: Save metadata
    print("\n[2/3] Saving preprocessing metadata...")
    metadata = save_preprocessing_metadata(processor, datasets, OUTPUT_DIR)

    # Fix 3: Generate LaTeX table
    print("\n[3/3] Generating LaTeX statistics table...")
    stats, latex_table = generate_statistics_table_for_paper(processor, datasets)

    print("\n" + "=" * 80)
    print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
    print("=" * 80)
    print("\nYou can now run enhanced_training.py with these files:")
    print(f"  • {OUTPUT_DIR}/binary_balanced.csv")
    print(f"  • {OUTPUT_DIR}/multiclass_balanced.csv")
    print(f"  • {OUTPUT_DIR}/scenario_balanced.csv")
    print("\n" + "=" * 80)

    return metadata, stats


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print(__doc__)
    print("\nThis file contains patches for data.py")
    print("Run data.py first, then import and use apply_fixes_to_data_processing()")
    print("\nExample:")
    print("  from data import main")
    print("  from data_fixes import apply_fixes_to_data_processing")
    print("  ")
    print("  processor, datasets, results = main()")
    print("  apply_fixes_to_data_processing(processor, datasets)")
