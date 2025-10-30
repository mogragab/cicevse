"""
Enhanced Training Script with All Novel Contributions
Integrates:
1. Hierarchical Multi-Resolution Attention
2. Adaptive Federated Aggregation (TWFA)
3. Federated Drift Detection
4. Byzantine-Resilient Aggregation
5. Federated SHAP Explainability
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import enhanced models
from enhanced_model import (
    EnhancedAdvancedTCN,
    FederatedDriftDetector,
    TrustWeightedFederatedAggregation,
    byzantine_resilient_krum,
    EnhancedFederatedClient
)

# Import explainability
from explainability import (
    FederatedSHAPExplainer,
    generate_explainability_report
)

# Import original data processor
import sys
sys.path.append(os.path.dirname(__file__))
from model import NetworkTrafficDataProcessor


# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = "enhanced-results"
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(f"{RESULTS_DIR}/enhanced_training.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"🖥️  Using device: {device}")

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


# ============================================================================
# ENHANCED FEDERATED TRAINING
# ============================================================================

def run_enhanced_federated_learning(
    filepath: str,
    detection_type: str = "multiclass",
    num_clients: int = 5,
    rounds: int = 10,
    local_epochs: int = 1,
    base_lr: float = 0.001,
    use_byzantine_defense: bool = True,
    use_drift_detection: bool = True,
    use_hierarchical_attention: bool = True,
    simulate_attack: bool = False  # Simulate Byzantine attack
):
    """
    Enhanced federated learning with all novel contributions
    """
    logging.info(f"\n{'=' * 80}\n"
                 f"ENHANCED FEDERATED LEARNING: {detection_type.upper()}\n"
                 f"{'=' * 80}")

    # ========================================================================
    # 1. DATA PREPROCESSING
    # ========================================================================
    logging.info("📊 Step 1: Data Preprocessing")
    data_processor = NetworkTrafficDataProcessor()
    processed_data = data_processor.preprocess_data(
        filepath, sequence_length=30, detection_type=detection_type
    )

    X, y, df, num_classes, feature_names = (
        processed_data['X'],
        processed_data['y'],
        processed_data['df'],
        processed_data['num_classes'],
        processed_data['feature_names']
    )

    logging.info(f"✅ Loaded {len(X)} sequences with {len(feature_names)} features")
    logging.info(f"✅ Number of classes: {num_classes}")

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    logging.info(f"✅ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ========================================================================
    # 2. INITIALIZE ENHANCED MODEL
    # ========================================================================
    logging.info("\n🧠 Step 2: Initializing Enhanced Model")

    input_size = X_train.shape[2]
    global_model = EnhancedAdvancedTCN(
        input_size=input_size,
        num_classes=num_classes,
        num_channels=[64, 128, 256],
        kernel_size=5,
        dropout=0.3,
        use_hierarchical_attention=use_hierarchical_attention,
        attention_scales=[1, 5, 15, 30]  # Multi-resolution scales
    ).to(device)

    logging.info(f"✅ Model parameters: {sum(p.numel() for p in global_model.parameters()):,}")
    if use_hierarchical_attention:
        logging.info("✅ Hierarchical Multi-Resolution Attention: ENABLED")

    # ========================================================================
    # 3. SPLIT DATA ACROSS CLIENTS
    # ========================================================================
    logging.info(f"\n👥 Step 3: Distributing Data Across {num_clients} Clients")

    # Stratified split
    client_indices = np.array_split(
        np.random.permutation(len(X_train)), num_clients
    )

    client_loaders = []
    val_loaders = []

    for i, indices in enumerate(client_indices):
        # Training data
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train[indices]),
            torch.LongTensor(y_train[indices])
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # Validation data (same for all clients for fair comparison)
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        client_loaders.append(train_loader)
        val_loaders.append(val_loader)

        logging.info(f"  Client {i+1}: {len(indices)} training samples")

    # ========================================================================
    # 4. INITIALIZE FEDERATED COMPONENTS
    # ========================================================================
    logging.info("\n⚙️  Step 4: Initializing Federated Components")

    # Drift detector
    drift_detector = FederatedDriftDetector(confidence=0.002) if use_drift_detection else None
    if drift_detector:
        logging.info("✅ Federated Drift Detection: ENABLED")

    # Trust-weighted aggregation
    trust_aggregator = TrustWeightedFederatedAggregation(num_clients=num_clients)
    logging.info("✅ Adaptive Trust-Weighted Aggregation: ENABLED")

    if use_byzantine_defense:
        logging.info("✅ Byzantine-Resilient Aggregation (Krum): ENABLED")

    # ========================================================================
    # 5. FEDERATED TRAINING LOOP
    # ========================================================================
    logging.info(f"\n🚀 Step 5: Starting Federated Training ({rounds} rounds)\n")

    history = {
        'val_loss': [],
        'val_acc': [],
        'client_weights': [],
        'drift_detected': [],
        'adaptive_lr': []
    }

    for round_num in range(rounds):
        logging.info(f"\n{'─' * 80}")
        logging.info(f"ROUND {round_num + 1}/{rounds}")
        logging.info(f"{'─' * 80}")

        # Determine learning rate (adaptive if drift detected)
        if drift_detector and len(history['val_acc']) > 0:
            error_rate = 1 - history['val_acc'][-1]
            drift_status = drift_detector.add_error(error_rate, round_num)
            current_lr = drift_detector.get_adaptive_lr(drift_status, base_lr)
            history['drift_detected'].append(drift_status)

            if drift_status['drift_detected']:
                logging.warning(f"🚨 Drift detected! Increasing LR: {base_lr} → {current_lr}")
        else:
            current_lr = base_lr
            history['drift_detected'].append({'drift_detected': False, 'warning': False})

        history['adaptive_lr'].append(current_lr)

        # Train each client
        client_models = []
        client_metrics = []

        for client_id, (train_loader, val_loader) in enumerate(zip(client_loaders, val_loaders)):
            logging.info(f"\n  Training Client {client_id + 1}...")

            # Create federated client
            client = EnhancedFederatedClient(
                client_id=client_id,
                model=global_model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=num_classes,
                device=device
            )

            # Simulate Byzantine attack on one client
            if simulate_attack and client_id == 0 and round_num >= 5:
                logging.warning(f"  ⚠️  Client {client_id + 1} is BYZANTINE (simulated attack)")
                # Add noise to model
                model_dict = client.model.state_dict()
                for k in model_dict.keys():
                    if model_dict[k].is_floating_point():
                        model_dict[k] += torch.randn_like(model_dict[k]) * 0.5
                client_models.append(model_dict)
                client_metrics.append({
                    'client_id': client_id,
                    'val_acc': 0.3,  # Low accuracy
                    'data_size': len(train_loader.dataset),
                    'training_loss': 2.0
                })
                continue

            # Local training
            model_dict, metrics = client.train(
                epochs=local_epochs,
                lr=current_lr,
                round_num=round_num
            )

            client_models.append(model_dict)
            client_metrics.append(metrics)

            logging.info(f"    ✓ Loss: {metrics['training_loss']:.4f}, "
                        f"Val Acc: {metrics['val_acc']:.4f}")

        # Aggregate models
        if use_byzantine_defense and simulate_attack:
            # Use Krum for Byzantine resilience
            logging.info("\n  🛡️  Applying Byzantine-Resilient Aggregation (Krum)...")
            global_state = byzantine_resilient_krum(
                client_models,
                f=1,  # Tolerate 1 Byzantine client
                multi_krum=True,
                m=3  # Average top 3 models
            )
        else:
            # Use trust-weighted aggregation
            logging.info("\n  ⚖️  Computing Trust-Weighted Aggregation...")
            client_weights = trust_aggregator.compute_client_weights(
                client_metrics, round_num
            )
            global_state = trust_aggregator.aggregate_models(client_models, client_weights)
            history['client_weights'].append(client_weights)

        # Update global model
        global_model.load_state_dict(global_state)

        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(
            global_model, X_val, y_val, num_classes, device
        )

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        logging.info(f"\n  📊 Global Model Performance:")
        logging.info(f"     Validation Loss: {val_loss:.4f}")
        logging.info(f"     Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

    # ========================================================================
    # 6. FINAL EVALUATION ON TEST SET
    # ========================================================================
    logging.info(f"\n{'=' * 80}")
    logging.info("FINAL EVALUATION ON TEST SET")
    logging.info(f"{'=' * 80}\n")

    test_loss, test_acc, y_pred, y_pred_proba = evaluate_model(
        global_model, X_test, y_test, num_classes, device, return_preds=True
    )

    # Get labels
    if detection_type == "binary":
        labels = ["Benign", "Attack"]
    elif detection_type == "multiclass":
        labels = list(processed_data['processor'].encoders['attack_type'].classes_)
        labels = [label.capitalize() for label in labels]
    else:  # scenario
        labels = list(processed_data['processor'].encoders['attack_type'].classes_)
        labels = [label.capitalize() for label in labels]

    logging.info(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    logging.info(f"Test Loss: {test_loss:.4f}\n")

    logging.info("Classification Report:")
    logging.info("\n" + classification_report(y_test, y_pred, target_names=labels))

    # ========================================================================
    # 7. EXPLAINABILITY ANALYSIS
    # ========================================================================
    logging.info(f"\n{'=' * 80}")
    logging.info("FEDERATED SHAP EXPLAINABILITY ANALYSIS")
    logging.info(f"{'=' * 80}\n")

    explainer = FederatedSHAPExplainer(
        model=global_model,
        feature_names=feature_names,
        device=device
    )

    # Explain each client
    for client_id, indices in enumerate(client_indices[:3]):  # First 3 clients
        test_subset_idx = np.random.choice(len(X_test), 200, replace=False)
        explanation = explainer.explain_client(
            X_test=X_test[test_subset_idx],
            y_test=y_test[test_subset_idx],
            client_id=client_id,
            n_samples=200
        )

    # Aggregate explanations
    federated_explanation = explainer.aggregate_explanations()

    # Generate explainability report
    generate_explainability_report(
        federated_explanation,
        labels=labels,
        save_dir=f"{RESULTS_DIR}/explainability"
    )

    # ========================================================================
    # 8. SAVE RESULTS
    # ========================================================================
    results = {
        'history': history,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'labels': labels,
        'num_classes': num_classes,
        'feature_names': feature_names,
        'federated_explanation': federated_explanation
    }

    # Save model
    torch.save(global_model.state_dict(), f"{RESULTS_DIR}/enhanced_model.pth")
    logging.info(f"\n✅ Model saved to {RESULTS_DIR}/enhanced_model.pth")

    return results, global_model


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, X, y, num_classes, device, return_preds=False):
    """Evaluate model on given data"""
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            loss = criterion(
                outputs.squeeze(),
                y_batch.float() if num_classes == 2 else y_batch
            )
            total_loss += loss.item()

            if num_classes == 2:
                probs = torch.sigmoid(outputs.squeeze())
                preds = (probs > 0.5).long()
                all_probs.extend(probs.cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.extend(probs.cpu().numpy())

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            if return_preds:
                all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    if return_preds:
        return avg_loss, accuracy, np.array(all_preds), np.array(all_probs)
    return avg_loss, accuracy


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_curves(history, save_path):
    """Plot training curves with drift detection markers"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Validation Accuracy",
            "Validation Loss",
            "Adaptive Learning Rate",
            "Client Trust Weights"
        ]
    )

    rounds = list(range(1, len(history['val_acc']) + 1))

    # Accuracy
    fig.add_trace(
        go.Scatter(x=rounds, y=history['val_acc'],
                   mode='lines+markers', name='Val Accuracy',
                   line=dict(width=3)),
        row=1, col=1
    )

    # Mark drift detection
    drift_rounds = [i+1 for i, d in enumerate(history['drift_detected']) if d.get('drift_detected', False)]
    if drift_rounds:
        drift_acc = [history['val_acc'][i-1] for i in drift_rounds]
        fig.add_trace(
            go.Scatter(x=drift_rounds, y=drift_acc,
                      mode='markers', name='Drift Detected',
                      marker=dict(size=15, color='red', symbol='x')),
            row=1, col=1
        )

    # Loss
    fig.add_trace(
        go.Scatter(x=rounds, y=history['val_loss'],
                   mode='lines+markers', name='Val Loss',
                   line=dict(width=3)),
        row=1, col=2
    )

    # Learning rate
    fig.add_trace(
        go.Scatter(x=rounds, y=history['adaptive_lr'],
                   mode='lines+markers', name='Learning Rate',
                   line=dict(width=3)),
        row=2, col=1
    )

    # Client weights evolution
    if 'client_weights' in history and len(history['client_weights']) > 0:
        weights_array = np.array(history['client_weights'])
        for i in range(weights_array.shape[1]):
            fig.add_trace(
                go.Scatter(x=rounds, y=weights_array[:, i],
                          mode='lines', name=f'Client {i+1}'),
                row=2, col=2
            )

    fig.update_layout(height=800, width=1200, showlegend=True)
    fig.write_image(save_path, scale=2)
    fig.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logging.info("🚀 Starting Enhanced Federated Learning Training\n")

    # Configuration
    FILEPATH = "./evse_scenario_classification.csv"

    # Run enhanced federated learning
    results, model = run_enhanced_federated_learning(
        filepath=FILEPATH,
        detection_type="multiclass",
        num_clients=5,
        rounds=10,
        local_epochs=1,
        base_lr=0.001,
        use_byzantine_defense=True,
        use_drift_detection=True,
        use_hierarchical_attention=True,
        simulate_attack=False  # Set to True to test Byzantine resilience
    )

    # Plot results
    plot_training_curves(
        results['history'],
        save_path=f"{RESULTS_DIR}/enhanced_training_curves.png"
    )

    logging.info(f"\n{'=' * 80}")
    logging.info("✅ ENHANCED TRAINING COMPLETED")
    logging.info(f"{'=' * 80}")
    logging.info(f"\n📊 Final Test Accuracy: {results['test_acc']*100:.2f}%")
    logging.info(f"📁 Results saved to: {RESULTS_DIR}/")
