"""
Federated SHAP Explainability Framework
Novel Contribution: Privacy-preserving explainability for federated IDS
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple
import shap
from captum.attr import IntegratedGradients, DeepLift, GradientShap


# ============================================================================
# FEDERATED SHAP EXPLAINER
# ============================================================================

class FederatedSHAPExplainer:
    """
    Privacy-preserving SHAP explanations for federated learning
    Computes local SHAP values per client, aggregates without sharing raw data
    """

    def __init__(self, model, feature_names, device='cpu'):
        self.model = model.to(device)
        self.feature_names = feature_names
        self.device = device
        self.client_explanations = []

    def explain_client(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        client_id: int,
        n_samples: int = 500,
        background_size: int = 100
    ) -> Dict:
        """
        Compute SHAP values for a single client (local computation)

        Args:
            X_test: Test data [n_samples, seq_len, features]
            y_test: True labels
            client_id: Client identifier
            n_samples: Number of samples to explain
            background_size: Background dataset size for SHAP

        Returns:
            Dictionary with aggregated SHAP values (privacy-preserving)
        """
        self.model.eval()

        # Select subset for explanation
        indices = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
        X_explain = torch.FloatTensor(X_test[indices]).to(self.device)
        y_explain = y_test[indices]

        # Background dataset for SHAP
        bg_indices = np.random.choice(len(X_test), min(background_size, len(X_test)), replace=False)
        X_background = torch.FloatTensor(X_test[bg_indices]).to(self.device)

        logging.info(f"🔍 Computing SHAP explanations for Client {client_id}...")

        # Use GradientShap (efficient for deep models)
        explainer = GradientShap(self.model)

        # Compute attributions (SHAP values)
        # Average across temporal dimension to get feature-level importance
        with torch.no_grad():
            baseline = torch.zeros_like(X_background)
            attributions = explainer.attribute(
                X_explain,
                baselines=baseline,
                target=None,
                n_samples=50
            )

        # Aggregate attributions: [n_samples, seq_len, features] -> [features]
        shap_values = attributions.cpu().numpy()

        # Global feature importance (average absolute SHAP across all samples and time)
        global_importance = np.abs(shap_values).mean(axis=(0, 1))

        # Per-class feature importance
        class_importance = {}
        for class_idx in np.unique(y_explain):
            class_mask = y_explain == class_idx
            class_shap = np.abs(shap_values[class_mask]).mean(axis=(0, 1))
            class_importance[int(class_idx)] = class_shap

        # Top features
        top_k = 20
        top_indices = np.argsort(global_importance)[-top_k:][::-1]
        top_features = [self.feature_names[i] for i in top_indices]
        top_importance = global_importance[top_indices]

        # Privacy-preserving summary (only share aggregated statistics)
        explanation = {
            'client_id': client_id,
            'global_importance': global_importance,
            'class_importance': class_importance,
            'top_features': top_features,
            'top_importance': top_importance,
            'top_indices': top_indices,
            'n_samples': n_samples
        }

        self.client_explanations.append(explanation)
        return explanation

    def aggregate_explanations(self, client_weights: np.ndarray = None) -> Dict:
        """
        Aggregate SHAP explanations across clients (federated aggregation)

        Args:
            client_weights: Optional weights for each client

        Returns:
            Aggregated global feature importance
        """
        if len(self.client_explanations) == 0:
            raise ValueError("No client explanations available")

        n_clients = len(self.client_explanations)

        if client_weights is None:
            client_weights = np.ones(n_clients) / n_clients
        else:
            client_weights = client_weights / client_weights.sum()

        # Weighted average of global importance
        global_importance = sum(
            w * exp['global_importance']
            for w, exp in zip(client_weights, self.client_explanations)
        )

        # Aggregate per-class importance
        all_classes = set()
        for exp in self.client_explanations:
            all_classes.update(exp['class_importance'].keys())

        class_importance = {}
        for class_idx in all_classes:
            class_imp = []
            for exp in self.client_explanations:
                if class_idx in exp['class_importance']:
                    class_imp.append(exp['class_importance'][class_idx])
            class_importance[class_idx] = np.mean(class_imp, axis=0)

        # Top features globally
        top_k = 20
        top_indices = np.argsort(global_importance)[-top_k:][::-1]
        top_features = [self.feature_names[i] for i in top_indices]
        top_importance = global_importance[top_indices]

        return {
            'global_importance': global_importance,
            'class_importance': class_importance,
            'top_features': top_features,
            'top_importance': top_importance,
            'top_indices': top_indices,
            'client_weights': client_weights
        }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_federated_shap_summary(
    federated_explanation: Dict,
    labels: List[str],
    save_path: str = None,
    title: str = "Federated SHAP Feature Importance"
):
    """
    Create comprehensive SHAP summary visualization
    """
    COLORS = ['#81C784', '#64B5F6', '#E45756', '#8B88C6']

    # Figure 1: Global Feature Importance (Bar chart)
    fig = go.Figure()

    top_features = federated_explanation['top_features']
    top_importance = federated_explanation['top_importance']

    fig.add_trace(go.Bar(
        y=top_features,
        x=top_importance,
        orientation='h',
        marker_color=COLORS[0],
        text=[f"{val:.4f}" for val in top_importance],
        textposition='auto',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Mean |SHAP Value| (Average Impact on Model Output)",
        yaxis_title="Features",
        height=600,
        width=800,
        template="plotly_white",
        yaxis={'autorange': 'reversed'},
        font=dict(size=12)
    )

    fig.show()
    if save_path:
        fig.write_image(f"{save_path}/federated_shap_global.png", scale=2)

    # Figure 2: Per-Class Feature Importance
    class_importance = federated_explanation['class_importance']

    if len(class_importance) > 1:
        fig2 = go.Figure()

        for class_idx, label in enumerate(labels):
            if class_idx in class_importance:
                imp = class_importance[class_idx]
                top_idx = federated_explanation['top_indices'][:10]

                fig2.add_trace(go.Bar(
                    name=label,
                    x=[federated_explanation['top_features'][i] for i in range(10)],
                    y=[imp[idx] for idx in top_idx],
                    marker_color=COLORS[class_idx % len(COLORS)]
                ))

        fig2.update_layout(
            title="Per-Class Feature Importance",
            xaxis_title="Features",
            yaxis_title="Mean |SHAP Value|",
            barmode='group',
            height=500,
            width=900,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_tickangle=-45
        )

        fig2.show()
        if save_path:
            fig2.write_image(f"{save_path}/federated_shap_per_class.png", scale=2)

    return fig


def plot_shap_waterfall(
    federated_explanation: Dict,
    attack_type: str,
    class_idx: int,
    save_path: str = None
):
    """
    Waterfall plot showing cumulative feature contribution for specific attack
    """
    class_imp = federated_explanation['class_importance'][class_idx]
    top_k = 15
    top_indices = np.argsort(class_imp)[-top_k:]

    features = [federated_explanation['top_features'][i] for i in range(min(top_k, len(federated_explanation['top_features'])))]
    values = [class_imp[idx] for idx in top_indices]

    # Waterfall chart
    fig = go.Figure(go.Waterfall(
        name=attack_type,
        orientation="h",
        y=features,
        x=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#E45756"}},
        increasing={"marker": {"color": "#81C784"}},
        textposition="outside"
    ))

    fig.update_layout(
        title=f"Feature Contribution Waterfall: {attack_type}",
        xaxis_title="SHAP Value",
        yaxis_title="Features",
        height=600,
        width=800,
        template="plotly_white",
        showlegend=False
    )

    fig.show()
    if save_path:
        fig.write_image(f"{save_path}/shap_waterfall_{attack_type.lower()}.png", scale=2)


def plot_client_explanation_variance(
    client_explanations: List[Dict],
    save_path: str = None
):
    """
    Visualize explanation variance across clients (shows data heterogeneity)
    """
    n_clients = len(client_explanations)
    n_features = len(client_explanations[0]['global_importance'])

    # Compute variance in feature importance across clients
    importance_matrix = np.array([
        exp['global_importance'] for exp in client_explanations
    ])

    mean_importance = importance_matrix.mean(axis=0)
    std_importance = importance_matrix.std(axis=0)
    cv = std_importance / (mean_importance + 1e-8)  # Coefficient of variation

    # Top features with highest variance
    top_k = 20
    top_var_idx = np.argsort(cv)[-top_k:][::-1]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=[f"Feature {i}" for i in top_var_idx],
        x=cv[top_var_idx],
        orientation='h',
        marker_color='#64B5F6',
        text=[f"{val:.3f}" for val in cv[top_var_idx]],
        textposition='auto'
    ))

    fig.update_layout(
        title="Feature Importance Variance Across Clients<br><sub>High variance indicates data heterogeneity</sub>",
        xaxis_title="Coefficient of Variation (σ/μ)",
        yaxis_title="Features",
        height=600,
        width=800,
        template="plotly_white",
        yaxis={'autorange': 'reversed'}
    )

    fig.show()
    if save_path:
        fig.write_image(f"{save_path}/client_explanation_variance.png", scale=2)


# ============================================================================
# INTEGRATED GRADIENTS (Alternative Explainability Method)
# ============================================================================

class FederatedIntegratedGradients:
    """
    Alternative to SHAP: Integrated Gradients for attribution
    Often faster and more stable for TCNs
    """

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.ig = IntegratedGradients(model)

    def explain(
        self,
        X: torch.Tensor,
        target: int = None,
        n_steps: int = 50
    ) -> np.ndarray:
        """
        Compute Integrated Gradients attributions

        Args:
            X: Input tensor [batch, seq_len, features]
            target: Target class (None for predicted class)
            n_steps: Number of integration steps

        Returns:
            Attributions [batch, seq_len, features]
        """
        self.model.eval()
        X = X.to(self.device)

        # Baseline (zero input)
        baseline = torch.zeros_like(X)

        # Compute attributions
        with torch.no_grad():
            attributions = self.ig.attribute(
                X,
                baselines=baseline,
                target=target,
                n_steps=n_steps
            )

        return attributions.cpu().numpy()


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_explainability_report(
    federated_explanation: Dict,
    labels: List[str],
    save_dir: str = "explainability_results"
):
    """
    Generate comprehensive explainability report
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Generate all visualizations
    plot_federated_shap_summary(
        federated_explanation,
        labels,
        save_path=save_dir,
        title="Federated SHAP: Global Feature Importance"
    )

    # Per-attack waterfall plots
    for class_idx, label in enumerate(labels):
        if class_idx in federated_explanation['class_importance']:
            plot_shap_waterfall(
                federated_explanation,
                attack_type=label,
                class_idx=class_idx,
                save_path=save_dir
            )

    # Generate text report
    report_path = f"{save_dir}/explainability_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FEDERATED EXPLAINABILITY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("TOP 10 MOST IMPORTANT FEATURES GLOBALLY:\n")
        f.write("-" * 80 + "\n")
        for i, (feat, imp) in enumerate(zip(
            federated_explanation['top_features'][:10],
            federated_explanation['top_importance'][:10]
        ), 1):
            f.write(f"{i:2d}. {feat:40s} | Importance: {imp:.6f}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("PER-CLASS FEATURE IMPORTANCE:\n")
        f.write("=" * 80 + "\n\n")

        for class_idx, label in enumerate(labels):
            if class_idx in federated_explanation['class_importance']:
                f.write(f"\n{label.upper()}:\n")
                f.write("-" * 80 + "\n")

                class_imp = federated_explanation['class_importance'][class_idx]
                top_idx = np.argsort(class_imp)[-10:][::-1]

                for i, idx in enumerate(top_idx, 1):
                    feat_name = federated_explanation['top_features'][i-1] if i <= len(federated_explanation['top_features']) else f"Feature_{idx}"
                    f.write(f"{i:2d}. {feat_name:40s} | Importance: {class_imp[idx]:.6f}\n")

    logging.info(f"✅ Explainability report saved to {save_dir}")


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'FederatedSHAPExplainer',
    'FederatedIntegratedGradients',
    'plot_federated_shap_summary',
    'plot_shap_waterfall',
    'plot_client_explanation_variance',
    'generate_explainability_report'
]
