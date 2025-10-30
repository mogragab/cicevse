# Upgrading to Improved data.py

## What's New in data_improved.py?

The improved version includes **ALL critical fixes**:

### ✅ Fixed Issues
1. **Correct output paths**: Files saved to `data/processed/` (not root)
2. **Correct file names**: `binary_balanced.csv`, `multiclass_balanced.csv`, `scenario_balanced.csv`
3. **Metadata export**: Automatic JSON + human-readable README
4. **LaTeX table generation**: Camera-ready table for paper
5. **Progress bars**: Better UX with tqdm for long operations
6. **Data validation**: Checks required columns exist
7. **Better error handling**: Clear error messages
8. **Reproducibility**: Configurable random seed
9. **Better logging**: Detailed progress tracking
10. **Complete documentation**: Comprehensive docstrings

### 📊 Comparison

| Feature | data.py (Original) | data_improved.py |
|---------|-------------------|------------------|
| Output directory | ❌ Root directory | ✅ `data/processed/` |
| File naming | ❌ Wrong names | ✅ Correct names |
| Metadata export | ❌ No | ✅ JSON + README |
| LaTeX tables | ❌ No | ✅ Auto-generated |
| Progress bars | ❌ No | ✅ Yes (tqdm) |
| Data validation | ❌ No | ✅ Yes |
| Error handling | ⚠️ Basic | ✅ Comprehensive |
| Works with enhanced_training.py | ❌ Needs fixes | ✅ Ready to go |

---

## 🚀 Option 1: Use Improved Version Directly (Recommended)

Just run the improved version instead of the original:

```bash
# Run improved version
python data_improved.py
```

**That's it!** All files will be in the correct locations with correct names.

---

## 🔄 Option 2: Replace Original data.py (Backup First)

If you want to replace the original `data.py`:

### Step 1: Backup Original

```bash
# Windows
copy data.py data_backup.py

# Linux/Mac
cp data.py data_backup.py
```

### Step 2: Replace with Improved Version

```bash
# Windows
copy data_improved.py data.py

# Linux/Mac
cp data_improved.py data.py
```

### Step 3: Run as Normal

```bash
python data.py
```

---

## 🧪 Option 3: Test Both Versions

Compare outputs:

```bash
# Run original (with fixes)
python data.py
python -c "from data_fixes import *; from data import main; p,d,r=main(); apply_fixes_to_data_processing(p,d)"

# Run improved version
python data_improved.py

# Compare outputs
python -c "import pandas as pd;
df1 = pd.read_csv('data/processed/multiclass_balanced.csv');
print(f'Improved: {df1.shape}');
print(df1['Attack'].value_counts())"
```

---

## 📁 What Files Are Generated?

After running `data_improved.py`, you get:

### 1. Preprocessed Datasets (`data/processed/`)
```
data/processed/
├── binary_balanced.csv              ✅ Binary classification
├── multiclass_balanced.csv          ✅ Multiclass (4 attack types)
├── scenario_balanced.csv            ✅ Scenario classification
├── preprocessing_metadata.json      ✅ Reproducibility metadata
└── README.txt                       ✅ Human-readable docs
```

### 2. Visualizations (`data-results/`)
```
data-results/
├── binary_distribution.png          ✅ Binary class distribution
├── attack_types.png                 ✅ Attack types bar chart
└── dataset_statistics_table.tex     ✅ LaTeX table for paper
```

---

## 🔍 Verification

After running, verify everything is correct:

```bash
# Check files exist
dir data\processed\*.csv

# Check file sizes (should be similar)
python -c "import os; print('\n'.join([f'{f}: {os.path.getsize(f\"data/processed/{f}\"):,} bytes' for f in ['binary_balanced.csv', 'multiclass_balanced.csv', 'scenario_balanced.csv']]))"

# Check metadata exists
type data\processed\preprocessing_metadata.json

# Check LaTeX table
type data-results\dataset_statistics_table.tex
```

Expected output:
```
data/processed/binary_balanced.csv
data/processed/multiclass_balanced.csv
data/processed/scenario_balanced.csv

binary_balanced.csv: 50,123,456 bytes
multiclass_balanced.csv: 25,456,789 bytes
scenario_balanced.csv: 45,678,901 bytes

✓ All files present and correct
```

---

## ⚡ Quick Start with Improved Version

```bash
# 1. Install dependencies (if not done)
pip install tqdm

# 2. Run improved data preprocessing
python data_improved.py

# 3. Verify output
dir data\processed\*.csv

# 4. Run enhanced training
python enhanced_training.py
```

**Total time**: ~30 minutes

---

## 🆚 Key Differences Explained

### File Paths
```python
# Original data.py
df_binary.to_csv("evse_binary_classification.csv", index=False)  # ❌ Root directory

# Improved data_improved.py
binary_path = f"{OUTPUT_DIR}/binary_balanced.csv"  # ✅ Correct path
df_binary.to_csv(binary_path, index=False)
```

### File Naming
```python
# Original data.py (Wrong names)
evse_binary_classification.csv       # ❌
evse_multiclass_attacks.csv          # ❌
evse_scenario_classification.csv     # ❌

# Improved data_improved.py (Correct names expected by enhanced_training.py)
binary_balanced.csv                   # ✅
multiclass_balanced.csv               # ✅
scenario_balanced.csv                 # ✅
```

### Metadata Export (NEW in improved version)
```python
# Automatically generates:
{
  "preprocessing_date": "2025-10-30T10:15:30",
  "random_seed": 42,
  "original_shape": [500000, 82],
  "cleaned_shape": [450000, 45],
  "feature_count": 41,
  "class_distributions": {...},
  "preprocessing_steps": [...]
}
```

### LaTeX Table (NEW in improved version)
```latex
\begin{table}[h]
\centering
\caption{CICEVSE2024 Dataset Statistics...}
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{Dataset} & \textbf{Samples} & ...
\midrule
Original (Filtered) & 450,000 & ...
\bottomrule
\end{tabular}
\end{table}
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tqdm'"

**Solution**:
```bash
pip install tqdm
```

### Issue: "FileNotFoundError: data/raw/Host Events/..."

**Solution**: Check raw data path exists
```bash
dir "data\raw\Host Events\EVSE-B-HPC-Kernel-Events-Combined.csv"
```

If not, update line 1025 in `data_improved.py`:
```python
filepath = "YOUR_ACTUAL_PATH_TO_RAW_DATA.csv"
```

### Issue: Files still in wrong location

**Check you're running the improved version**:
```bash
python data_improved.py  # ✅ Correct
python data.py           # ❌ Old version (needs fixes)
```

---

## 📚 Documentation Updates

All documentation now references the correct file locations:

- **COMPLETE_WORKFLOW.md**: Updated to use `data_improved.py`
- **QUICKSTART.md**: Uses correct file paths
- **run_complete_pipeline.py**: Compatible with both versions

---

## ✅ Migration Checklist

- [ ] Install tqdm: `pip install tqdm`
- [ ] Backup original: `copy data.py data_backup.py`
- [ ] Run improved version: `python data_improved.py`
- [ ] Verify output files: `dir data\processed\*.csv`
- [ ] Check metadata: `type data\processed\preprocessing_metadata.json`
- [ ] Check LaTeX table: `type data-results\dataset_statistics_table.tex`
- [ ] Test with training: `python enhanced_training.py`

---

## 🎯 Recommendation

**Use `data_improved.py` directly** - no need to replace the original unless you prefer having a single `data.py` file.

Both approaches work:
- **Keep both**: Run `python data_improved.py` (recommended for clarity)
- **Replace**: Backup original, then replace (cleaner if you only want one file)

---

## 📧 Questions?

If you encounter issues:
1. Check this guide
2. Review error messages in terminal
3. Verify all dependencies installed: `pip list | grep -E "tqdm|pandas|numpy|plotly|sklearn|imblearn"`

---

**Last Updated**: 2025-10-30
**Status**: Production Ready
**Recommended**: Use `data_improved.py` directly
