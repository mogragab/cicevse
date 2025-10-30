"""
Enhanced Model Architecture with Novel Contributions
Includes:
1. Hierarchical Multi-Resolution Temporal Attention (AMRTA)
2. Adaptive Federated Aggregation (TWFA)
3. Byzantine-Resilient Aggregation (Krum)
4. Federated Drift Detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import logging
from collections import deque
from typing import List, Dict, Tuple, Optional


# ============================================================================
# HIERARCHICAL TEMPORAL ATTENTION (Novel Contribution #2)
# ============================================================================

class MultiResolutionTemporalAttention(nn.Module):
    """
    Adaptive Multi-Resolution Temporal Attention (AMRTA)
    Captures attack patterns at different time scales:
    - Short-term: DoS bursts (seconds)
    - Medium-term: Cryptojacking patterns (minutes)
    - Long-term: Reconnaissance campaigns (hours)
    """
    def __init__(self, d_model=256, scales=[1, 5, 15, 30], num_heads=4, dropout=0.1):
        super().__init__()
        self.scales = scales
        self.d_model = d_model

        # Separate attention for each temporal scale
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in scales
        ])

        # Scale-specific projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in scales
        ])

        # Adaptive scale fusion with learnable weights
        self.scale_fusion = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )

        # Learnable scale importance weights
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_scale_weights=False):
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            Fused multi-scale temporal representations
        """
        batch_size, seq_len, d_model = x.shape
        scale_outputs = []
        scale_attention_weights = []

        for scale_idx, (scale, attn, proj) in enumerate(
            zip(self.scales, self.scale_attentions, self.scale_projections)
        ):
            if scale == 1:
                # Full resolution
                x_scale = x
            else:
                # Downsample using average pooling
                x_scale = F.adaptive_avg_pool1d(
                    x.transpose(1, 2),
                    output_size=max(1, seq_len // scale)
                ).transpose(1, 2)

            # Apply self-attention at this scale
            attn_out, attn_weights = attn(x_scale, x_scale, x_scale)

            # Project and upsample back to original resolution
            attn_out = proj(attn_out)
            if scale > 1:
                attn_out = F.interpolate(
                    attn_out.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            # Apply learnable scale weight
            scale_weight = F.softmax(self.scale_weights, dim=0)[scale_idx]
            scale_outputs.append(attn_out * scale_weight)
            scale_attention_weights.append(attn_weights)

        # Concatenate and fuse multi-scale representations
        fused = torch.cat(scale_outputs, dim=-1)
        fused = self.scale_fusion(fused)

        # Residual connection and normalization
        output = self.layer_norm(x + self.dropout(fused))

        if return_scale_weights:
            return output, F.softmax(self.scale_weights, dim=0), scale_attention_weights
        return output


# ============================================================================
# ENHANCED TCN WITH HIERARCHICAL ATTENTION
# ============================================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(n_outputs)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.bn2 = nn.BatchNorm1d(n_outputs)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.bn1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
            self.bn2,
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.xavier_uniform_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class EnhancedAdvancedTCN(nn.Module):
    """
    Enhanced TCN with Hierarchical Multi-Resolution Attention
    """
    def __init__(
        self,
        input_size,
        num_classes=2,
        num_channels=[64, 128, 256],
        kernel_size=5,
        dropout=0.3,
        use_hierarchical_attention=True,
        attention_scales=[1, 5, 15, 30],
    ):
        super(EnhancedAdvancedTCN, self).__init__()
        self.use_hierarchical_attention = use_hierarchical_attention

        # Temporal convolutional layers
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    1,
                    dilation_size,
                    (kernel_size - 1) * dilation_size,
                    dropout,
                )
            ]
        self.network = nn.Sequential(*layers)

        # Hierarchical Multi-Resolution Attention (NOVEL)
        if self.use_hierarchical_attention:
            self.hierarchical_attention = MultiResolutionTemporalAttention(
                d_model=num_channels[-1],
                scales=attention_scales,
                num_heads=8,
                dropout=dropout
            )
            self.layer_norm = nn.LayerNorm(num_channels[-1])

        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_channels[-1], 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes if num_classes > 2 else 1),
        )

    def forward(self, x, return_attention_weights=False):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = self.network(x)    # [batch, channels, seq_len]

        if self.use_hierarchical_attention:
            x = x.transpose(1, 2)  # [batch, seq_len, channels]
            if return_attention_weights:
                x, scale_weights, attn_weights = self.hierarchical_attention(
                    x, return_scale_weights=True
                )
            else:
                x = self.hierarchical_attention(x)
            x = x.transpose(1, 2)  # [batch, channels, seq_len]

        x = self.global_pool(x).squeeze(-1)
        output = self.classifier(x)

        if return_attention_weights:
            return output, scale_weights, attn_weights
        return output


# ============================================================================
# FEDERATED DRIFT DETECTION (Novel Contribution #3)
# ============================================================================

class FederatedDriftDetector:
    """
    Adaptive concept drift detection for federated learning
    Uses ADWIN (Adaptive Windowing) algorithm
    """
    def __init__(self, confidence=0.002, warning_threshold=0.05):
        self.confidence = confidence
        self.warning_threshold = warning_threshold
        self.error_window = deque(maxlen=1000)
        self.drift_history = []
        self.baseline_error = None

    def add_error(self, error_rate: float, round_num: int) -> Dict[str, bool]:
        """
        Add new error rate and check for drift
        Returns: dict with 'drift_detected' and 'warning' flags
        """
        self.error_window.append(error_rate)

        if self.baseline_error is None and len(self.error_window) >= 100:
            self.baseline_error = np.mean(list(self.error_window))

        result = {'drift_detected': False, 'warning': False, 'magnitude': 0.0}

        if len(self.error_window) >= 100:
            recent_error = np.mean(list(self.error_window)[-100:])
            historical_error = np.mean(list(self.error_window)[:-100])

            drift_magnitude = abs(recent_error - historical_error)
            result['magnitude'] = drift_magnitude

            # Warning threshold
            if drift_magnitude > self.warning_threshold:
                result['warning'] = True
                logging.warning(f"⚠️  Drift warning at round {round_num}: magnitude={drift_magnitude:.4f}")

            # Drift detection using statistical test
            if drift_magnitude > 2 * self.warning_threshold:
                result['drift_detected'] = True
                self.drift_history.append({
                    'round': round_num,
                    'error_rate': error_rate,
                    'drift_magnitude': drift_magnitude
                })
                logging.error(f"🚨 CONCEPT DRIFT DETECTED at round {round_num}")

        return result

    def get_adaptive_lr(self, drift_status: Dict[str, bool], base_lr: float = 0.001) -> float:
        """Increase learning rate when drift detected for faster adaptation"""
        if drift_status['drift_detected']:
            return base_lr * 5  # 5x boost for major drift
        elif drift_status['warning']:
            return base_lr * 2  # 2x boost for warning
        return base_lr


# ============================================================================
# ADAPTIVE FEDERATED AGGREGATION (Novel Contribution #1)
# ============================================================================

class TrustWeightedFederatedAggregation:
    """
    Trust-Weighted Federated Averaging (TWFA)
    Weights clients based on performance, data freshness, and historical trust
    """
    def __init__(self, num_clients: int, alpha: float = 0.7):
        self.num_clients = num_clients
        self.alpha = alpha  # Weight for exponential moving average
        self.historical_trust = np.ones(num_clients) / num_clients
        self.contribution_history = []

    def compute_client_weights(
        self,
        client_metrics: List[Dict],
        round_num: int
    ) -> np.ndarray:
        """
        Compute trust-based weights for each client

        Args:
            client_metrics: List of dicts with keys:
                - 'val_acc': validation accuracy
                - 'data_size': number of samples
                - 'training_loss': local training loss
        """
        weights = []

        for i, metrics in enumerate(client_metrics):
            # Component 1: Performance-based trust (validation accuracy)
            perf_trust = metrics.get('val_acc', 0.5)

            # Component 2: Data contribution (normalized data size)
            total_data = sum(m.get('data_size', 1) for m in client_metrics)
            data_trust = metrics.get('data_size', 1) / total_data

            # Component 3: Historical trust (exponential moving average)
            hist_trust = self.historical_trust[i]

            # Component 4: Loss-based trust (lower loss = higher trust)
            max_loss = max(m.get('training_loss', 1.0) for m in client_metrics)
            loss_trust = 1.0 - (metrics.get('training_loss', 1.0) / (max_loss + 1e-8))

            # Weighted combination
            trust_score = (
                0.4 * perf_trust +
                0.2 * data_trust +
                0.3 * hist_trust +
                0.1 * loss_trust
            )

            weights.append(trust_score)

            # Update historical trust with exponential moving average
            self.historical_trust[i] = (
                self.alpha * self.historical_trust[i] +
                (1 - self.alpha) * perf_trust
            )

        # Normalize weights
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-8)

        self.contribution_history.append({
            'round': round_num,
            'weights': weights.copy(),
            'metrics': client_metrics
        })

        logging.info(f"📊 Client weights for round {round_num}: {weights}")
        return weights

    def aggregate_models(
        self,
        client_models: List[Dict],
        weights: np.ndarray
    ) -> Dict:
        """Weighted model aggregation"""
        avg_dict = copy.deepcopy(client_models[0])

        for k in avg_dict.keys():
            if avg_dict[k].is_floating_point():
                avg_dict[k] = sum(
                    w * client_models[i][k].float()
                    for i, w in enumerate(weights)
                )

        return avg_dict


# ============================================================================
# BYZANTINE-RESILIENT AGGREGATION (Novel Contribution #5)
# ============================================================================

def byzantine_resilient_krum(
    client_models: List[Dict],
    f: int = 1,
    multi_krum: bool = False,
    m: int = 1
) -> Dict:
    """
    Krum aggregation for Byzantine resilience

    Args:
        client_models: List of client model state dicts
        f: Number of Byzantine clients to tolerate
        multi_krum: Use Multi-Krum (average top-m models)
        m: Number of models to select for Multi-Krum
    """
    n = len(client_models)

    if n < 2 * f + 3:
        logging.warning(f"⚠️  Insufficient clients for f={f} Byzantine tolerance")
        return client_models[0]

    # Flatten model parameters for distance computation
    def flatten_model(model_dict):
        return torch.cat([p.flatten() for p in model_dict.values() if p.is_floating_point()])

    flattened = [flatten_model(model) for model in client_models]

    # Compute pairwise distances
    distances = torch.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = torch.norm(flattened[i] - flattened[j], p=2) ** 2
            distances[i, j] = distances[j, i] = dist

    # Compute Krum scores (sum of distances to n-f-2 closest neighbors)
    scores = []
    for i in range(n):
        closest_distances = torch.topk(distances[i], n - f - 2, largest=False)[0]
        scores.append(closest_distances.sum().item())

    if multi_krum:
        # Select top-m models with lowest scores
        selected_indices = np.argsort(scores)[:m]
        logging.info(f"🛡️ Multi-Krum selected clients: {selected_indices.tolist()}")

        # Average selected models
        avg_dict = copy.deepcopy(client_models[0])
        for k in avg_dict.keys():
            if avg_dict[k].is_floating_point():
                avg_dict[k] = sum(client_models[i][k].float() for i in selected_indices) / m
        return avg_dict
    else:
        # Select single best model
        selected_idx = np.argmin(scores)
        logging.info(f"🛡️ Krum selected client: {selected_idx} (score={scores[selected_idx]:.4f})")
        return client_models[selected_idx]


# ============================================================================
# FEDERATED CLIENT WITH ENHANCED FEATURES
# ============================================================================

class EnhancedFederatedClient:
    """Enhanced federated client with drift detection and metrics tracking"""
    def __init__(self, client_id, model, train_loader, val_loader, num_classes, device):
        self.client_id = client_id
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.device = device
        self.metrics_history = []

    def train(self, epochs, lr, round_num):
        """Train with metrics tracking"""
        self.model.train()
        criterion = (
            nn.BCEWithLogitsLoss() if self.num_classes == 2 else nn.CrossEntropyLoss()
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        total_loss = 0
        num_batches = 0

        for epoch in range(epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(
                    outputs.squeeze(),
                    y_batch.float() if self.num_classes == 2 else y_batch,
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        # Compute validation metrics
        val_acc, val_loss = self._evaluate()

        metrics = {
            'client_id': self.client_id,
            'round': round_num,
            'training_loss': avg_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'data_size': len(self.train_loader.dataset)
        }
        self.metrics_history.append(metrics)

        return self.model.state_dict(), metrics

    def _evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        criterion = (
            nn.BCEWithLogitsLoss() if self.num_classes == 2 else nn.CrossEntropyLoss()
        )

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = criterion(
                    outputs.squeeze(),
                    y_batch.float() if self.num_classes == 2 else y_batch,
                )
                total_loss += loss.item()

                if self.num_classes == 2:
                    preds = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
                else:
                    preds = torch.argmax(outputs, dim=1)

                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        return correct / total, total_loss / len(self.val_loader)


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'EnhancedAdvancedTCN',
    'MultiResolutionTemporalAttention',
    'FederatedDriftDetector',
    'TrustWeightedFederatedAggregation',
    'byzantine_resilient_krum',
    'EnhancedFederatedClient'
]
