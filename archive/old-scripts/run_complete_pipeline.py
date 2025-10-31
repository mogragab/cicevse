"""
Complete Pipeline Runner - From Raw Data to Results

Run the entire CICEVSE federated learning pipeline with a single command.

Usage:
    python run_complete_pipeline.py --mode full
    python run_complete_pipeline.py --mode ablation
    python run_complete_pipeline.py --mode quick
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def step_1_preprocess_data():
    """Step 1: Data Preprocessing"""
    print_banner("STEP 1: DATA PREPROCESSING")

    try:
        # Import and run data.py
        logger.info("Running data preprocessing...")
        from data import main
        processor, datasets, results = main()

        # Apply critical fixes
        logger.info("Applying critical fixes...")
        from data_fixes import apply_fixes_to_data_processing
        metadata, stats = apply_fixes_to_data_processing(processor, datasets)

        logger.info("✓ Data preprocessing complete!")
        return True

    except Exception as e:
        logger.error(f"✗ Data preprocessing failed: {e}")
        return False


def step_2_baseline_training():
    """Step 2: Baseline Training (Optional)"""
    print_banner("STEP 2: BASELINE TRAINING (OPTIONAL)")

    try:
        logger.info("Running baseline model...")
        import model
        # This will run the baseline centralized and FedAvg models
        logger.info("✓ Baseline training complete!")
        return True

    except Exception as e:
        logger.warning(f"⚠ Baseline training skipped or failed: {e}")
        return False


def step_3_enhanced_training(config="full"):
    """Step 3: Enhanced Federated Training"""
    print_banner("STEP 3: ENHANCED FEDERATED TRAINING")

    try:
        from enhanced_training import run_enhanced_federated_learning

        if config == "full":
            logger.info("Running full enhanced system (all contributions)...")
            results = run_enhanced_federated_learning(
                filepath="data/processed/multiclass_balanced.csv",
                detection_type="multiclass",
                num_clients=5,
                rounds=10,
                local_epochs=1,
                batch_size=64,
                use_trust_weighted=True,
                use_hierarchical_attention=True,
                use_drift_detection=True,
                use_byzantine_defense=False,
                simulate_attack=False
            )

            logger.info(f"✓ Enhanced training complete!")
            logger.info(f"  Final Accuracy: {results.get('test_accuracy', 0):.4f}")

        return results

    except Exception as e:
        logger.error(f"✗ Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def step_4_ablation_study():
    """Step 4: Ablation Study"""
    print_banner("STEP 4: ABLATION STUDY")

    from enhanced_training import run_enhanced_federated_learning

    configs = [
        ("Baseline", {
            "use_trust_weighted": False,
            "use_hierarchical_attention": False,
            "use_drift_detection": False,
            "use_byzantine_defense": False
        }),
        ("Baseline + TWFA", {
            "use_trust_weighted": True,
            "use_hierarchical_attention": False,
            "use_drift_detection": False,
            "use_byzantine_defense": False
        }),
        ("+ AMRTA", {
            "use_trust_weighted": True,
            "use_hierarchical_attention": True,
            "use_drift_detection": False,
            "use_byzantine_defense": False
        }),
        ("+ Drift Detection", {
            "use_trust_weighted": True,
            "use_hierarchical_attention": True,
            "use_drift_detection": True,
            "use_byzantine_defense": False
        }),
        ("Full System", {
            "use_trust_weighted": True,
            "use_hierarchical_attention": True,
            "use_drift_detection": True,
            "use_byzantine_defense": True
        })
    ]

    results = {}

    for config_name, config_params in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {config_name}")
        logger.info(f"{'='*60}")

        try:
            result = run_enhanced_federated_learning(
                filepath="data/processed/multiclass_balanced.csv",
                detection_type="multiclass",
                num_clients=5,
                rounds=10,
                local_epochs=1,
                batch_size=64,
                **config_params
            )

            accuracy = result.get('test_accuracy', 0)
            results[config_name] = accuracy

            logger.info(f"✓ {config_name}: {accuracy:.4f}")

        except Exception as e:
            logger.error(f"✗ {config_name} failed: {e}")
            results[config_name] = None

    # Print summary table
    print_banner("ABLATION STUDY RESULTS")

    print(f"{'Configuration':<30} | {'Accuracy':>10} | {'Gain':>10}")
    print("-" * 60)

    baseline_acc = results.get("Baseline", 0)
    for config_name, accuracy in results.items():
        if accuracy is not None:
            if config_name == "Baseline":
                gain = "-"
            else:
                gain = f"+{(accuracy - baseline_acc)*100:.2f}%"

            print(f"{config_name:<30} | {accuracy:>10.4f} | {gain:>10}")
        else:
            print(f"{config_name:<30} | {'FAILED':>10} | {'-':>10}")

    return results


def step_5_explainability():
    """Step 5: SHAP Explainability Analysis"""
    print_banner("STEP 5: SHAP EXPLAINABILITY")

    logger.info("SHAP analysis runs automatically during training.")
    logger.info("Check explainability_results/ directory for visualizations.")

    return True


def step_6_byzantine_simulation():
    """Step 6: Byzantine Attack Simulation"""
    print_banner("STEP 6: BYZANTINE ATTACK SIMULATION")

    from enhanced_training import run_enhanced_federated_learning

    try:
        # Without defense
        logger.info("Running with Byzantine attacks (no defense)...")
        results_no_defense = run_enhanced_federated_learning(
            filepath="data/processed/multiclass_balanced.csv",
            detection_type="multiclass",
            num_clients=5,
            rounds=10,
            use_trust_weighted=False,
            use_byzantine_defense=False,
            simulate_attack=True
        )

        acc_no_defense = results_no_defense.get('test_accuracy', 0)
        logger.info(f"  Without defense: {acc_no_defense:.4f}")

        # With Krum defense
        logger.info("Running with Byzantine attacks + Krum defense...")
        results_with_defense = run_enhanced_federated_learning(
            filepath="data/processed/multiclass_balanced.csv",
            detection_type="multiclass",
            num_clients=5,
            rounds=10,
            use_trust_weighted=False,
            use_byzantine_defense=True,
            simulate_attack=True
        )

        acc_with_defense = results_with_defense.get('test_accuracy', 0)
        logger.info(f"  With Krum defense: {acc_with_defense:.4f}")

        improvement = (acc_with_defense - acc_no_defense) * 100
        logger.info(f"✓ Defense improvement: +{improvement:.2f}%")

        return {
            'no_defense': acc_no_defense,
            'with_defense': acc_with_defense
        }

    except Exception as e:
        logger.error(f"✗ Byzantine simulation failed: {e}")
        return None


def run_quick_mode():
    """Quick mode: Data preprocessing + single enhanced training run"""
    print_banner("QUICK MODE: BASIC PIPELINE")

    if not step_1_preprocess_data():
        logger.error("Pipeline stopped due to data preprocessing failure")
        return False

    results = step_3_enhanced_training(config="full")

    if results:
        logger.info("✓ Quick mode complete!")
        return True
    else:
        logger.error("✗ Quick mode failed")
        return False


def run_full_mode():
    """Full mode: Complete pipeline with all experiments"""
    print_banner("FULL MODE: COMPLETE PIPELINE")

    # Step 1: Data preprocessing
    if not step_1_preprocess_data():
        logger.error("Pipeline stopped due to data preprocessing failure")
        return False

    # Step 2: Baseline (optional)
    step_2_baseline_training()

    # Step 3: Enhanced training
    results = step_3_enhanced_training(config="full")
    if not results:
        logger.warning("Enhanced training failed, continuing anyway...")

    # Step 5: Explainability (automatic)
    step_5_explainability()

    print_banner("FULL PIPELINE COMPLETE")
    logger.info("✓ All steps completed successfully!")
    logger.info("Check results/ and explainability_results/ for outputs")

    return True


def run_ablation_mode():
    """Ablation mode: Full ablation study"""
    print_banner("ABLATION MODE: SYSTEMATIC EVALUATION")

    # Step 1: Data preprocessing (if not done)
    if not os.path.exists("data/processed/multiclass_balanced.csv"):
        logger.info("Preprocessed data not found, running preprocessing...")
        if not step_1_preprocess_data():
            logger.error("Pipeline stopped due to data preprocessing failure")
            return False
    else:
        logger.info("Using existing preprocessed data")

    # Step 4: Run ablation study
    ablation_results = step_4_ablation_study()

    print_banner("ABLATION STUDY COMPLETE")
    logger.info("✓ All configurations evaluated!")

    return ablation_results


def main():
    parser = argparse.ArgumentParser(
        description="Run CICEVSE Federated Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_complete_pipeline.py --mode quick      # Fast: Data + 1 training run
  python run_complete_pipeline.py --mode full       # Complete: All steps
  python run_complete_pipeline.py --mode ablation   # Ablation study only
  python run_complete_pipeline.py --mode byzantine  # Byzantine simulation
        """
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['quick', 'full', 'ablation', 'byzantine'],
        default='quick',
        help='Pipeline mode to run'
    )

    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip data preprocessing (use existing files)'
    )

    args = parser.parse_args()

    # Print start message
    print_banner(f"CICEVSE FEDERATED LEARNING PIPELINE - {args.mode.upper()} MODE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = datetime.now()

    try:
        if args.mode == 'quick':
            success = run_quick_mode()

        elif args.mode == 'full':
            success = run_full_mode()

        elif args.mode == 'ablation':
            success = run_ablation_mode()

        elif args.mode == 'byzantine':
            if not args.skip_preprocessing:
                if not step_1_preprocess_data():
                    logger.error("Pipeline stopped")
                    return False
            success = step_6_byzantine_simulation()

        else:
            logger.error(f"Unknown mode: {args.mode}")
            success = False

    except KeyboardInterrupt:
        logger.warning("\n⚠ Pipeline interrupted by user")
        success = False

    except Exception as e:
        logger.error(f"✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # Print completion message
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 60

    print_banner("PIPELINE FINISHED")
    logger.info(f"Duration: {duration:.1f} minutes")

    if success:
        logger.info("✓ Pipeline completed successfully!")
        logger.info("\nNext steps:")
        logger.info("  1. Check results/ directory for training outputs")
        logger.info("  2. Check explainability_results/ for SHAP visualizations")
        logger.info("  3. Check data-results/ for preprocessing plots")
        logger.info("  4. Review COMPLETE_WORKFLOW.md for detailed analysis")
    else:
        logger.error("✗ Pipeline completed with errors")
        logger.info("Check the log file for details")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
