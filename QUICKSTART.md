# Quick Start Guide - CICEVSE Federated Learning

Get from raw data to research results in 3 commands.

---

## ⚡ Super Quick Start (5 minutes to first results)

```bash
# 1. Install dependencies
uv pip install -e .

# 2. Run complete pipeline (quick mode)
python run_complete_pipeline.py --mode quick

# 3. Check results
ls results/
ls explainability_results/
```

---

## 🎯 What You Get

After running the quick mode, you'll have:

✅ **Preprocessed Datasets**:
- `data/processed/binary_balanced.csv`
- `data/processed/multiclass_balanced.csv`
- `data/processed/scenario_balanced.csv`

✅ **Training Results** (~98.40% accuracy):
- `results/training_curves.png`
- `results/confusion_matrix.png`
- `results/training_log.txt`

✅ **SHAP Explanations**:
- `explainability_results/federated_shap_global.png`
- `explainability_results/shap_waterfall_*.png`
- `explainability_results/explainability_report.txt`

---

## 📋 Pipeline Modes

### Mode 1: Quick (30 minutes)
Fast path to first results - runs data preprocessing + one enhanced training run.

```bash
python run_complete_pipeline.py --mode quick
```

### Mode 2: Full (1-2 hours)
Complete pipeline with all components (no ablation study).

```bash
python run_complete_pipeline.py --mode full
```

### Mode 3: Ablation Study (2-3 hours)
Systematic evaluation of all 5 contributions.

```bash
python run_complete_pipeline.py --mode ablation
```

### Mode 4: Byzantine Simulation (30-45 minutes)
Test robustness against malicious clients.

```bash
python run_complete_pipeline.py --mode byzantine
```

---

## 🔧 Manual Step-by-Step

If you prefer manual control:

### Step 1: Data Preprocessing
```python
# Run data.py
python data.py

# Apply fixes
from data import main
from data_fixes import apply_fixes_to_data_processing

processor, datasets, results = main()
apply_fixes_to_data_processing(processor, datasets)
```

### Step 2: Enhanced Training
```python
from enhanced_training import run_enhanced_federated_learning

results = run_enhanced_federated_learning(
    filepath="data/processed/multiclass_balanced.csv",
    detection_type="multiclass",
    num_clients=5,
    rounds=10,
    use_trust_weighted=True,           # TWFA
    use_hierarchical_attention=True,   # AMRTA
    use_drift_detection=True,          # Drift Detection
    use_byzantine_defense=False        # Krum (optional)
)

print(f"Final Accuracy: {results['test_accuracy']:.4f}")
```

---

## 📊 Expected Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 98.40% |
| **Precision** | 98.35% |
| **Recall** | 98.40% |
| **F1-Score** | 98.37% |

**Improvements**:
- vs. Standard FedAvg: **+3.28%**
- vs. Centralized: **+1.05%**

---

## 🐛 Common Issues

### "File not found: data/processed/*.csv"
**Solution**: Run data preprocessing first
```bash
python data.py
python -c "from data_fixes import *; from data import main; p,d,r=main(); apply_fixes_to_data_processing(p,d)"
```

### "CUDA out of memory"
**Solution**: Reduce batch size
```python
run_enhanced_federated_learning(batch_size=32, ...)
```

### "ImportError: No module named 'shap'"
**Solution**: Install missing dependencies
```bash
pip install shap captum --upgrade
```

---

## 📖 Detailed Documentation

- **`COMPLETE_WORKFLOW.md`** - Full 6-step pipeline with all experiments
- **`IMPROVEMENTS_SUMMARY.md`** - Details on 5 novel contributions
- **`DATA_PY_IMPROVEMENTS.md`** - Data preprocessing enhancements
- **`CLAUDE.md`** - Repository structure and common issues

---

## ✅ Verification Checklist

After running quick mode:

```bash
# Check preprocessed data exists
ls data/processed/*.csv

# Check training completed
cat results/training_log.txt | grep "Final"

# Check explainability generated
ls explainability_results/*.png

# View accuracy
cat results/training_log.txt | grep "Test Accuracy"
```

---

## 🚀 For Your Paper

### Generate All Results
```bash
# Run full ablation study (2-3 hours)
python run_complete_pipeline.py --mode ablation

# Results will be printed in LaTeX-ready format
```

### Copy to Paper
1. **Dataset statistics**: `data-results/dataset_statistics_table.tex`
2. **Training curves**: `results/training_curves.png`
3. **SHAP plots**: `explainability_results/*.png`
4. **Ablation results**: Check terminal output from ablation mode

---

## 💡 Pro Tips

1. **GPU is highly recommended** (8GB+ VRAM)
2. **Use quick mode first** to verify everything works
3. **Run ablation study overnight** (takes 2-3 hours)
4. **Save logs** - they contain reproducibility information
5. **Check visualizations** in `results/` and `explainability_results/`

---

## 📧 Need Help?

1. Check error log: `pipeline_run_*.log`
2. Review `COMPLETE_WORKFLOW.md` for detailed instructions
3. Check `CLAUDE.md` for repository-specific guidance

---

**Last Updated**: 2025-10-30
**Estimated Time (Quick Mode)**: 30 minutes
**Estimated Time (Full Mode)**: 1-2 hours
**Estimated Time (Ablation)**: 2-3 hours
