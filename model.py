import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots

# Define your custom color list
COLORS = [
    "#81C784",  # Light Green
    "#64B5F6",  # Light Blue
    "#E45756",  # Coral Red
    # "#7FD2BC",  # Teal Green
    "#8B88C6",  # Lavender Gray
    "#6DCCDA",  # Sky Blue
    # "#FF9DA7",  # Soft Pink
    "#FF8A65",  # Light Burnt Orange
    "#4DD0E1",  # Light Cyan
    "#F06292",  # Light Pink
    "#90A4AE",  # Light Slate Gray
    "#AED581",  # Light Lime Green
    "#A1887F",  # Light Sepia Brown
    "#E6EE9C",  # Light Lime Yellow
    "#9575CD",  # Medium Purple
    "#FFB74D",  # Light Orange
    "#4FC3F7",  # Sky Blue
    "#EF5350",  # Light Red
]

custom_template = go.layout.Template(
    layout=go.Layout(
        template="plotly_white",
        colorway=COLORS,
        margin=dict(l=40, r=40, b=40, t=40),
        font=dict(
            family="Arial",
            size=14,
            color="darkslategrey",
        ),
        width=600,
        height=600,
        autosize=False,
    ),
    data=dict(
        scatter=[dict(line=dict(width=8))]
    ),
)

# Set as default template
pio.templates["my_custom"] = custom_template
pio.templates.default = "my_custom"

LAYOUT = go.Layout(
    template="plotly_white",
    colorway=COLORS,
    margin=dict(l=40, r=40, b=40, t=40),
    font=dict(
        family="Arial",
        size=14,
        color="darkslategrey",
    ),
    width=600,
    height=600,
    autosize=False,
)
import warnings

warnings.filterwarnings("ignore")

import os
import shutil

MODEL_RESULTS = "model-results"

# Create results folder if it doesn't exist, or clean it if it does
if os.path.exists(MODEL_RESULTS):
    # Remove all contents of the folder
    shutil.rmtree(MODEL_RESULTS)

# Create the results folder
os.makedirs(MODEL_RESULTS)
print(f"Results folder {MODEL_RESULTS} is ready for new outputs")
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    average_precision_score,
    precision_score,
    recall_score,
    auc,
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import copy
import logging
import sys
import re

# --- Setup ---
warnings.filterwarnings("ignore")
pio.templates.default = "plotly_white"


# Setup logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{MODEL_RESULTS}/analysis_log.txt", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        return torch.matmul(attn_probs, V)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        Q = (
            self.W_q(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.W_k(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.W_v(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )
        return self.W_o(attn_output)


class AdvancedTCN(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes=2,
        num_channels=[64, 128, 256],
        kernel_size=5,
        dropout=0.3,
        use_attention=True,
    ):
        super(AdvancedTCN, self).__init__()
        self.use_attention = use_attention
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

        if self.use_attention:
            self.attention = MultiHeadAttention(num_channels[-1], 8, dropout)
            self.layer_norm = nn.LayerNorm(num_channels[-1])

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

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        if self.use_attention:
            x = x.transpose(1, 2)
            x = self.layer_norm(x + self.attention(x))
            x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)


# --- Data Processor & Federated Learning ---
class NetworkTrafficDataProcessor:
    def __init__(self):
        self.encoders = {}

    def preprocess_data(self, filepath, sequence_length=30, detection_type="binary"):
        """
        Advanced preprocessing function for EVSE datasets with robust error handling
        and comprehensive data validation.

        Args:
            filepath (str): Path to the EVSE CSV dataset
            sequence_length (int): Length of temporal sequences to create
            detection_type (str): Type of detection - 'binary', 'multiclass', or 'scenario'

        Returns:
            dict: Preprocessed data with features, targets, and metadata
        """
        logging.info(
            f"Preprocessing EVSE data for {detection_type} detection from {filepath}..."
        )

        # Load dataset with error handling
        try:
            df = pd.read_csv(filepath)
            logging.info(f"Successfully loaded dataset with shape: {df.shape}")
            logging.info(f"Columns: {list(df.columns)}")
        except FileNotFoundError:
            logging.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        except pd.errors.EmptyDataError:
            logging.error(f"Empty CSV file: {filepath}")
            raise ValueError(f"CSV file is empty: {filepath}")
        except Exception as e:
            logging.error(f"Error loading file {filepath}: {e}")
            raise RuntimeError(f"Failed to load dataset: {e}")

        # Validate dataset is not empty
        if df.empty:
            raise ValueError("Dataset is empty after loading")

        logging.info(f"Dataset info: {df.shape[0]} rows, {df.shape[1]} columns")
        logging.info(
            f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB"
        )

        # Handle different detection types and their corresponding target columns
        if detection_type == "binary":
            if "Label" not in df.columns:
                available_cols = list(df.columns)
                logging.error(
                    f"Binary dataset must have 'Label' column. Available columns: {available_cols}"
                )
                raise ValueError("Binary dataset must have 'Label' column")

            target_col = "Label"
            df["target"] = df[target_col].astype(int)
            df["attack_type"] = df[target_col].map({0: "Benign", 1: "Attack"})
            num_classes = 2

            # Validate binary values
            unique_labels = df["target"].unique()
            if not all(label in [0, 1] for label in unique_labels):
                logging.error(f"Binary labels must be 0 or 1, found: {unique_labels}")
                raise ValueError(f"Invalid binary labels: {unique_labels}")

        elif detection_type == "multiclass":
            if "Attack" not in df.columns:
                available_cols = list(df.columns)
                logging.error(
                    f"Multiclass dataset must have 'Attack' column. Available columns: {available_cols}"
                )
                raise ValueError("Multiclass dataset must have 'Attack' column")

            target_col = "Attack"
            df["attack_type"] = df[target_col].astype(str).str.lower().str.strip()

            # Remove any null or empty attack types
            before_cleanup = len(df)
            df = df[
                df["attack_type"].notna()
                & (df["attack_type"] != "")
                & (df["attack_type"] != "nan")
            ]
            after_cleanup = len(df)

            if after_cleanup < before_cleanup:
                logging.info(
                    f"Removed {before_cleanup - after_cleanup} rows with invalid attack types"
                )

            if df.empty:
                raise ValueError("No valid attack type data remaining after cleanup")

            le = LabelEncoder()
            df["target"] = le.fit_transform(df["attack_type"])
            self.encoders["attack_type"] = le
            num_classes = len(le.classes_)

            logging.info(f"Attack types found: {list(le.classes_)}")

        elif detection_type == "scenario":
            if "Scenario" not in df.columns:
                available_cols = list(df.columns)
                logging.error(
                    f"Scenario dataset must have 'Scenario' column. Available columns: {available_cols}"
                )
                raise ValueError("Scenario dataset must have 'Scenario' column")

            target_col = "Scenario"
            df["attack_type"] = df[target_col].astype(str).str.lower().str.strip()

            # Remove any null or empty scenarios
            before_cleanup = len(df)
            df = df[
                df["attack_type"].notna()
                & (df["attack_type"] != "")
                & (df["attack_type"] != "nan")
            ]
            after_cleanup = len(df)

            if after_cleanup < before_cleanup:
                logging.info(
                    f"Removed {before_cleanup - after_cleanup} rows with invalid scenarios"
                )

            if df.empty:
                raise ValueError("No valid scenario data remaining after cleanup")

            le = LabelEncoder()
            df["target"] = le.fit_transform(df["attack_type"])
            self.encoders["attack_type"] = le
            num_classes = len(le.classes_)

            logging.info(f"Scenarios found: {list(le.classes_)}")

        else:
            raise ValueError(
                f"Unknown detection_type: {detection_type}. Must be 'binary', 'multiclass', or 'scenario'"
            )

        # Log class distribution
        class_distribution = df["attack_type"].value_counts()
        logging.info(f"Class distribution:")
        for class_name, count in class_distribution.items():
            percentage = (count / len(df)) * 100
            logging.info(f"  {class_name}: {count:,} ({percentage:.1f}%)")

        # Identify and validate feature columns
        exclude_cols = [target_col, "target", "attack_type"]

        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        logging.info(f"Found {len(numeric_cols)} numeric columns")

        # Advanced feature filtering with multiple criteria
        feature_names = []
        feature_stats = []

        for col in numeric_cols:
            if col in exclude_cols:
                continue

            col_data = df[col]

            # Check 1: Must have more than 1 unique value (variance check)
            unique_count = col_data.nunique()
            if unique_count <= 1:
                logging.debug(f"Skipping {col}: only {unique_count} unique values")
                continue

            # Check 2: Must not be all zeros
            if (col_data == 0).all():
                logging.debug(f"Skipping {col}: all values are zero")
                continue

            # Check 3: Must not be all NaN
            if col_data.isnull().all():
                logging.debug(f"Skipping {col}: all values are NaN")
                continue

            # Check 4: Must have reasonable variance (not constant-like)
            if col_data.dtype in ["float64", "int64"]:
                try:
                    variance = col_data.var()
                    if variance == 0 or pd.isna(variance):
                        logging.debug(f"Skipping {col}: zero or NaN variance")
                        continue
                except:
                    logging.debug(f"Skipping {col}: cannot calculate variance")
                    continue

            # Check 5: Must not have excessive missing values (>95%)
            missing_ratio = col_data.isnull().sum() / len(col_data)
            if missing_ratio > 0.95:
                logging.debug(f"Skipping {col}: {missing_ratio:.1%} missing values")
                continue

            # Feature passed all checks
            feature_names.append(col)
            feature_stats.append(
                {
                    "name": col,
                    "unique_values": unique_count,
                    "missing_ratio": missing_ratio,
                    "variance": (
                        col_data.var()
                        if col_data.dtype in ["float64", "int64"]
                        else None
                    ),
                }
            )

        if len(feature_names) == 0:
            logging.error("No valid features found after filtering")
            logging.error(f"Available numeric columns: {numeric_cols}")
            logging.error(f"Excluded columns: {exclude_cols}")
            raise ValueError(
                "No valid features found - all numeric columns failed quality checks"
            )

        logging.info(
            f"Selected {len(feature_names)} features from {len(numeric_cols)} numeric columns"
        )
        logging.info(f"Feature selection summary:")
        logging.info(f"  - Excluded columns: {len(exclude_cols)}")
        logging.info(
            f"  - Failed quality checks: {len(numeric_cols) - len(exclude_cols) - len(feature_names)}"
        )
        logging.info(f"  - Valid features: {len(feature_names)}")

        self.feature_names = feature_names

        # Extract and validate feature data
        try:
            X = df[feature_names].values.astype(np.float32)
            logging.info(f"Extracted feature matrix with shape: {X.shape}")
        except Exception as e:
            logging.error(f"Failed to extract features: {e}")
            raise ValueError(f"Feature extraction failed: {e}")

        # Comprehensive data cleaning
        original_shape = X.shape

        # Handle NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            nan_percentage = (nan_count / X.size) * 100
            logging.info(
                f"Found {nan_count:,} NaN values ({nan_percentage:.2f}% of data)"
            )
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            logging.info("Replaced NaN values with 0.0")

        # Handle infinite values
        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            inf_percentage = (inf_count / X.size) * 100
            logging.info(
                f"Found {inf_count:,} infinite values ({inf_percentage:.2f}% of data)"
            )
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            logging.info("Replaced infinite values with 0.0")

        # Feature scaling with robust fallback
        logging.info("Applying feature scaling...")
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logging.info("Applied StandardScaler successfully")

            # Verify scaling didn't introduce problems
            if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
                raise ValueError("StandardScaler produced NaN or infinite values")

        except Exception as e:
            logging.warning(
                f"StandardScaler failed ({e}), using RobustScaler as fallback"
            )
            try:
                from sklearn.preprocessing import RobustScaler

                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                logging.info("Applied RobustScaler successfully")
            except Exception as e2:
                logging.error(f"Both StandardScaler and RobustScaler failed: {e2}")
                logging.info("Using original features without scaling")
                X_scaled = X
                scaler = None

        # Extract and validate target values
        try:
            y = df["target"].values.astype(np.int64)
            logging.info(f"Extracted target vector with shape: {y.shape}")
        except Exception as e:
            logging.error(f"Failed to extract targets: {e}")
            raise ValueError(f"Target extraction failed: {e}")

        # Validate data consistency
        assert len(X_scaled) == len(
            y
        ), f"Feature-target length mismatch: {len(X_scaled)} vs {len(y)}"
        assert X_scaled.shape[1] == len(
            feature_names
        ), f"Feature count mismatch: {X_scaled.shape[1]} vs {len(feature_names)}"
        assert (
            y.min() >= 0 and y.max() < num_classes
        ), f"Target values out of range: {y.min()}-{y.max()} for {num_classes} classes"

        logging.info("Data consistency validation passed")

        # Create temporal sequences
        if len(X_scaled) < sequence_length:
            original_seq_length = sequence_length
            sequence_length = max(
                1, len(X_scaled) // 3
            )  # Use 1/3 of data as sequence length
            logging.warning(
                f"Dataset too small ({len(X_scaled)}) for sequence length {original_seq_length}"
            )
            logging.info(f"Adjusting sequence length to {sequence_length}")

        if sequence_length < 1:
            raise ValueError(
                f"Sequence length must be at least 1, got {sequence_length}"
            )

        logging.info(f"Creating temporal sequences with length {sequence_length}...")

        X_seq, y_seq = [], []
        for i in range(len(X_scaled) - sequence_length + 1):
            sequence = X_scaled[i : i + sequence_length]
            target = y[i + sequence_length - 1]  # Use the last label in the sequence

            # Validate sequence
            if np.isnan(sequence).any() or np.isinf(sequence).any():
                logging.warning(f"Invalid sequence at index {i}, skipping")
                continue

            X_seq.append(sequence)
            y_seq.append(target)

        if len(X_seq) == 0:
            raise ValueError(
                "No valid sequences created - all sequences contained invalid values"
            )

        # Convert to numpy arrays with proper data types
        X_seq = np.array(X_seq, dtype=np.float32)
        y_seq = np.array(y_seq, dtype=np.int64)

        # Final comprehensive validation
        assert not np.isnan(X_seq).any(), "NaN values found in final sequences"
        assert not np.isinf(X_seq).any(), "Infinite values found in final sequences"
        assert X_seq.shape[0] == len(
            y_seq
        ), f"Sequence count mismatch: {X_seq.shape[0]} vs {len(y_seq)}"
        assert (
            X_seq.shape[1] == sequence_length
        ), f"Sequence length mismatch: {X_seq.shape[1]} vs {sequence_length}"
        assert X_seq.shape[2] == len(
            feature_names
        ), f"Feature count mismatch: {X_seq.shape[2]} vs {len(feature_names)}"

        # Log final statistics
        final_class_distribution = np.bincount(y_seq, minlength=num_classes)
        logging.info(f"Successfully created {len(X_seq):,} temporal sequences")
        logging.info(f"Final sequence shape: {X_seq.shape}")
        logging.info(f"Final target shape: {y_seq.shape}")
        logging.info(
            f"Final class distribution: {dict(enumerate(final_class_distribution))}"
        )

        # Calculate data quality metrics
        data_quality = {
            "original_samples": len(df),
            "final_sequences": len(X_seq),
            "sequence_retention": len(X_seq) / len(df),
            "features_selected": len(feature_names),
            "features_total": len(numeric_cols),
            "feature_retention": (
                len(feature_names) / len(numeric_cols) if len(numeric_cols) > 0 else 0
            ),
            "nan_values_cleaned": nan_count,
            "inf_values_cleaned": inf_count,
            "scaling_method": type(scaler).__name__ if scaler else "None",
        }

        logging.info("Data quality summary:")
        for metric, value in data_quality.items():
            if isinstance(value, float) and 0 < value < 1:
                logging.info(f"  {metric}: {value:.1%}")
            else:
                logging.info(f"  {metric}: {value}")

        # Return comprehensive results
        return {
            "X": X_seq,
            "y": y_seq,
            "df": df,
            "num_classes": num_classes,
            "feature_names": feature_names,
            "processor": self,
            "sequence_length": sequence_length,
            "scaler": scaler,
            "data_quality": data_quality,
            "class_distribution": dict(enumerate(final_class_distribution)),
            "original_class_distribution": class_distribution.to_dict(),
        }


class FederatedClient:
    def __init__(self, model, train_loader, num_classes, device):
        self.model = copy.deepcopy(model).to(device)
        self.train_loader = train_loader
        self.num_classes = num_classes
        self.device = device

    def train(self, epochs, lr):
        self.model.train()
        criterion = (
            nn.BCEWithLogitsLoss() if self.num_classes == 2 else nn.CrossEntropyLoss()
        )
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for _ in range(epochs):
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
        return self.model.state_dict()


def federated_average(models):
    avg_dict = copy.deepcopy(models[0])
    for k in avg_dict.keys():
        if avg_dict[k].is_floating_point():
            avg_dict[k] = torch.stack([m[k].float() for m in models]).mean(0)
    return avg_dict


# --- Plotting Functions ---
def plot_individual_figure(figure, title):
    """
    Displays and saves a Plotly figure.
    - Applies a standard layout.
    - Saves the figure to the 'results' directory.
    """
    logging.info(f"Generating plot: {title}")

    # Sanitize title for filename
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title)

    # Apply the standard layout and a title
    figure.update_layout(LAYOUT)
    # figure.update_layout(title_text=f"<b>{title}</b>", title_x=0.5)

    # Show the figure
    figure.show()

    # Save the figure
    try:
        pio.write_image(figure, f"{MODEL_RESULTS}/{safe_title}.png", scale=2)
        logging.info(f"Successfully saved plot to: results/{safe_title}.png")
    except Exception as e:
        logging.error(f"Could not save plot {safe_title}.png. Error: {e}")


def plot_training_progress(history, title):
    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.add_trace(
    #     go.Scatter(
    #         x=list(range(len(history["val_loss"]))),
    #         y=history["val_loss"],
    #         name="Validation Loss",
    #         line=dict(color="blue"),
    #     ),
    #     secondary_y=False,
    # )
    # fig.add_trace(
    #     go.Scatter(
    #         x=list(range(len(history["val_acc"]))),
    #         y=history["val_acc"],
    #         name="Validation Accuracy",
    #         line=dict(color="red"),
    #     ),
    #     secondary_y=True,
    # )
    # fig.update_yaxes(title_text="Loss", secondary_y=False)
    # fig.update_yaxes(title_text="Accuracy", secondary_y=True)

    fig = go.Figure()

    # Add validation loss trace
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history["val_loss"]) + 1)),
            y=history["val_loss"],
            name="Validation Loss",
            # line=dict(color="blue"),
            line=dict(width=2.5),
            yaxis="y",
        )
    )

    # Add validation accuracy trace
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history["val_acc"]), +1)),
            y=history["val_acc"],
            name="Validation Accuracy",
            line=dict(width=2.5),
            yaxis="y2",
        )
    )

    # Update layout to create secondary y-axis
    fig.update_layout(
        yaxis=dict(title="Loss", side="left"),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right"),
        # legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
        # xaxis_range=[0, len(history["val_acc"])]
    )

    plot_individual_figure(fig, title)


def plot_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)

    fig = go.Figure(
        layout=LAYOUT,
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale=[
                [0, "#FFFFFF"],  # White for minimum values
                [1, "#81C784"],  # Your green for maximum values
            ],
            text=cm,
            texttemplate="%{text}",
            showscale=False,
        ),
    )
    fig.update_layout(
        xaxis_title="Predicted Label", yaxis_title="True Label", width=600, height=500
    )
    fig.show()

    fig.write_image(f"{MODEL_RESULTS}/{title}.png", scale=2)


def plot_roc_curve(y_true, y_pred_proba, labels, num_classes, title):
    fig = go.Figure()
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                # name=f"AUC = {auc(fpr, tpr):.3f}",
                name=f"AUC = {auc(fpr, tpr):.3f}",
                line=dict(width=2),
                fill="tozeroy",
            )
        )
    else:
        for i, label in enumerate(labels):
            y_true_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, y_pred_proba[:, i])
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    # name=f"{label} (AUC={auc(fpr, tpr):.2f})",
                    name=f"{label}",
                    showlegend=True,
                    line=dict(width=2),
                )
            )
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey")
    )
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(
            x=0.95,  # Position near right edge
            y=0.05,  # Position near bottom
            xanchor="right",  # Anchor legend's right edge
            yanchor="bottom",  # Anchor legend's bottom edge
        ),
    )
    plot_individual_figure(fig, title)


def plot_pr_curve(y_true, y_pred_proba, labels, num_classes, title):
    fig = go.Figure()
    if num_classes == 2:
        p, r, _ = precision_recall_curve(y_true, y_pred_proba)
        fig.add_trace(
            go.Scatter(
                x=r,
                y=p,
                name=f"AP = {average_precision_score(y_true, y_pred_proba):.3f}",
                # name=f"AP = {average_precision_score(y_true, y_pred_proba):.3f}",
                # fill="toself",
                showlegend=True,
                line=dict(width=2),
            )
        )
    else:
        for i, label in enumerate(labels):
            y_true_bin = (y_true == i).astype(int)
            p, r, _ = precision_recall_curve(y_true_bin, y_pred_proba[:, i])
            fig.add_trace(
                go.Scatter(
                    x=r,
                    y=p,
                    # name=f"{label} (AP={average_precision_score(y_true_bin, y_pred_proba[:, i]):.2f})",
                    name=f"{label}",
                    line=dict(width=2),
                    # fill="toself",
                )
            )
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend=dict(
            x=0.05,  # Position near left edge
            y=0.05,  # Position near bottom
            xanchor="left",  # Anchor legend's left edge
            yanchor="bottom",  # Anchor legend's bottom edge
        ),
    )
    plot_individual_figure(fig, title)


def plot_attack_distribution(df, title):
    counts = df["attack_type"].value_counts()
    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index, y=counts.values, text=counts.values, textposition="auto"
            )
        ]
    )
    fig.update_layout(xaxis_title="Attack Type", yaxis_title="Count")
    plot_individual_figure(fig, title)


def plot_feature_importance(feature_names, model, title):
    try:
        weights = model.network[0].conv1.weight.data.cpu().numpy()
        importances = np.abs(weights).mean(axis=(0, 2))
        df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(20)
        )

        fig = px.bar(
            df,
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "Importance Score", "feature": "Feature"},
            text="importance",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            uniformtext_minsize=8,
            uniformtext_mode="hide",
        )
        plot_individual_figure(fig, title)
    except Exception as e:
        logging.error(f"Could not plot feature importance: {e}")


def plot_pca_2d(X, y, labels, title):
    pca = PCA(n_components=2)
    X_flat = X.reshape(X.shape[0], -1)
    X_pca = pca.fit_transform(X_flat)

    fig = go.Figure()
    unique_labels = np.unique(y)
    for i, label_code in enumerate(unique_labels):
        label_name = str(labels[label_code])
        fig.add_trace(
            go.Scatter(
                x=X_pca[y == label_code, 0],
                y=X_pca[y == label_code, 1],
                mode="markers",
                name=label_name,
                marker=dict(size=5, opacity=0.7),
                showlegend=False,
            )
        )
    fig.update_layout(
        xaxis_title="Principal Component 1", yaxis_title="Principal Component 2"
    )
    plot_individual_figure(fig, title)


def plot_pca_3d(X, y, labels, title):
    pca = PCA(n_components=3)
    X_flat = X.reshape(X.shape[0], -1)
    X_pca = pca.fit_transform(X_flat)

    fig = go.Figure()
    unique_labels = np.unique(y)
    for i, label_code in enumerate(unique_labels):
        label_name = str(labels[label_code])
        fig.add_trace(
            go.Scatter3d(
                x=X_pca[y == label_code, 0],
                y=X_pca[y == label_code, 1],
                z=X_pca[y == label_code, 2],
                mode="markers",
                name=label_name,
                marker=dict(size=3, opacity=0.6),
                showlegend=False,
            )
        )
    fig.update_layout(
        scene=dict(
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            zaxis_title="PC 3",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Adjust these values to zoom out
            ),
        )
    )
    plot_individual_figure(fig, title)


def plot_class_distribution_per_client(client_data_split, y_train, labels, title):
    # client_dist_data = []
    # for i, client_indices in enumerate(client_data_split):
    #     client_y = y_train[client_indices]
    #     class_counts = pd.Series(client_y).value_counts()
    #     for class_idx, count in class_counts.items():
    #         client_dist_data.append(
    #             {
    #                 "Client": f"Client {i + 1}",
    #                 "Class": labels[class_idx],
    #                 "Count": count,
    #             }
    #         )

    # if not client_dist_data:
    #     logging.warning("No data to plot for client class distribution.")
    #     return

    # df_dist = pd.DataFrame(client_dist_data)
    # fig = px.bar(df_dist, x="Client", y="Count", color="Class", barmode="stack")

    client_dist_data = []
    for i, client_indices in enumerate(client_data_split):
        client_y = y_train[client_indices]
        class_counts = pd.Series(client_y).value_counts()
        for class_idx, count in class_counts.items():
            client_dist_data.append(
                {
                    "Client": f"Client {i + 1}",
                    "Class": labels[class_idx],
                    "Count": count,
                }
            )

    if not client_dist_data:
        logging.warning("No data to plot for client class distribution.")
        # return or handle empty data case
    else:
        df_dist = pd.DataFrame(client_dist_data)

        # Create figure with graph objects
        fig = go.Figure(layout=LAYOUT)

        # Get unique classes and clients
        unique_classes = df_dist["Class"].unique()
        unique_clients = df_dist["Client"].unique()

        # Add a bar trace for each class
        for class_name in unique_classes:
            class_data = df_dist[df_dist["Class"] == class_name]

            fig.add_trace(
                go.Bar(
                    x=class_data["Client"],
                    y=class_data["Count"],
                    name=class_name,
                    # Plotly will automatically assign different colors to each trace
                )
            )

        # Update layout for stacked bars
        fig.update_layout(
            barmode="stack",
            xaxis_title="Client",
            yaxis_title="Count",
            # title="Client Class Distribution",
            legend=dict(
                orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5
            ),
        )

    plot_individual_figure(fig, title)


def plot_prediction_probability_distribution(y_pred_proba, num_classes, title):
    probs = y_pred_proba if num_classes == 2 else np.max(y_pred_proba, axis=1)
    fig = go.Figure(data=[go.Histogram(x=probs, nbinsx=50, name="Probabilities")])
    fig.update_layout(xaxis_title="Predicted Probability", yaxis_title="Frequency")
    plot_individual_figure(fig, title)


def plot_metrics_per_class(y_true, y_pred, labels, title):
    report = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True
    )
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df = metrics_df[
        ~metrics_df.index.isin(["accuracy", "macro avg", "weighted avg"])
    ]
    fig = go.Figure()
    for metric in ["precision", "recall", "f1-score"]:
        fig.add_trace(
            go.Bar(name=metric.capitalize(), x=metrics_df.index, y=metrics_df[metric])
        )
    fig.update_layout(barmode="group", xaxis_title="Class", yaxis_title="Score")
    plot_individual_figure(fig, title)


def plot_sunburst_attacks(df, title):
    attack_counts = df["Is_Attack"].value_counts()
    benign_count = attack_counts.get(0, 0)
    total_attack_count = attack_counts.get(1, 0)
    attack_type_counts = df[df["Is_Attack"] == 1]["attack_type"].value_counts()

    labels = ["All Traffic", "Benign", "Attack"] + list(attack_type_counts.index)
    parents = ["", "All Traffic", "All Traffic"] + ["Attack"] * len(attack_type_counts)
    values = [
        benign_count + total_attack_count,
        benign_count,
        total_attack_count,
    ] + list(attack_type_counts.values)

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            hoverinfo="label+percent parent+value",
        )
    )
    fig.update_layout(margin=dict(t=50, l=10, r=10, b=10))
    plot_individual_figure(fig, title)


def plot_feature_correlation_heatmap(df, feature_names, title):
    corr = df[feature_names].corr()

    blue_gradient = [
        [0.0, "#FFFFFF"],  # White (minimum values)
        [0.2, "#E3F2FD"],  # Very light blue
        [0.4, "#BBDEFB"],  # Light blue
        [0.6, "#90CAF9"],  # Medium light blue
        [0.8, "#64B5F6"],  # Your custom blue
        [1.0, "#1976D2"],  # Darker blue (maximum values)
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale=blue_gradient,
            zmin=-1,
            zmax=1,
            showscale=False,
        )
    )
    plot_individual_figure(fig, title)


def plot_violin_probabilities(y_true, y_pred_proba, labels, title):
    fig = go.Figure()
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:  # Multiclass
        data = []
        for i, label in enumerate(labels):
            data.append(
                go.Violin(
                    y=y_pred_proba[y_true == i, i],
                    name=label,
                    box_visible=True,
                    meanline_visible=True,
                    showlegend=False,
                )
            )
        fig.add_traces(data)
    else:  # Binary
        fig.add_trace(
            go.Violin(
                y=y_pred_proba[y_true == 0],
                name=labels[0],
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Violin(
                y=y_pred_proba[y_true == 1],
                name=labels[1],
                box_visible=True,
                meanline_visible=True,
                showlegend=False,
            )
        )
    fig.update_layout(yaxis_title="Predicted Probability")
    plot_individual_figure(fig, title)


def plot_federated_vs_centralized(fed_history, cent_history, title):
    fig = go.Figure(layout=LAYOUT)
    fig.add_trace(
        go.Scatter(
            y=fed_history["val_acc"],
            name="Federated Accuracy",
            mode="lines+markers",
            line=dict(width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            y=cent_history["val_acc"],
            name="Centralized Accuracy",
            mode="lines+markers",
            line=dict(width=2),
        )
    )
    fig.update_layout(
        xaxis_title="Round / Epoch",
        yaxis_title="Validation Accuracy",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    plot_individual_figure(fig, title)


def plot_feature_distribution(df, feature_name, title):
    fig = go.Figure(layout=LAYOUT)
    for attack_type in df["attack_type"].unique():
        fig.add_trace(
            go.Histogram(
                x=df[df["attack_type"] == attack_type][feature_name],
                name=attack_type,
                opacity=0.75,
            )
        )
    fig.update_layout(barmode="overlay", xaxis_title=feature_name, yaxis_title="Count")
    plot_individual_figure(fig, title)


def plot_data_overview(df, title):
    num_samples = len(df)
    num_features = len(df.columns)
    attack_counts = df["attack_type"].value_counts()

    fig = go.Figure(
        layout=LAYOUT,
        data=[
            go.Table(
                header=dict(
                    values=["<b>Metric</b>", "<b>Value</b>"],
                    fill_color="paleturquoise",
                    align="left",
                ),
                cells=dict(
                    values=[
                        ["Total Samples", "Total Features", "Unique Attack Types"]
                        + list(attack_counts.index),
                        [f"{num_samples:,}", num_features, len(attack_counts)]
                        + [f"{count:,}" for count in attack_counts.values],
                    ],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ],
    )
    plot_individual_figure(fig, title)


def plot_parallel_coordinates(df, feature_names, title):
    sample_df = df.sample(n=min(500, len(df)), random_state=42)
    fig = px.parallel_coordinates(
        sample_df,
        dimensions=feature_names,
        color="target",
        labels={"target": "Attack Type"},
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    plot_individual_figure(fig, title)


def plot_anomaly_scores(y_true, anomaly_scores, labels, title):
    df = pd.DataFrame({"True Label": y_true, "Anomaly Score": anomaly_scores})
    df["Label Name"] = df["True Label"].apply(lambda x: labels[x])

    fig = go.Figure(layout=LAYOUT)
    for label_name in df["Label Name"].unique():
        fig.add_trace(
            go.Histogram(
                x=df[df["Label Name"] == label_name]["Anomaly Score"],
                name=label_name,
                opacity=0.75,
            )
        )

    fig.update_layout(
        barmode="overlay",
        xaxis_title="Anomaly Score (Lower is more anomalous)",
        yaxis_title="Count",
        legend=dict(orientation="h", yanchor="top", y=1.15, xanchor="center", x=0.5),
    )
    plot_individual_figure(fig, title)


# --- Main Analysis Runner ---
def run_analysis(
    filepath,
    detection_type="binary",
    federated=True,
    sequence_length=30,
    rounds=5,
    num_clients=5,
):
    logging.info(
        f"\n{'=' * 30}\nRunning {detection_type.upper()} {'Federated' if federated else 'Centralized'} Analysis\n{'=' * 30}"
    )

    data_processor = NetworkTrafficDataProcessor()
    processed_data = data_processor.preprocess_data(
        filepath, sequence_length, detection_type
    )
    X, y, df, num_classes, feature_names = (
        processed_data["X"],
        processed_data["y"],
        processed_data["df"],
        processed_data["num_classes"],
        processed_data["feature_names"],
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    input_size = X_train.shape[2]
    model = AdvancedTCN(input_size=input_size, num_classes=num_classes)

    # Anomaly Detection
    logging.info("Training Isolation Forest for anomaly detection...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    iso_forest = IsolationForest(contamination="auto", random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_flat)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    anomaly_scores = iso_forest.decision_function(X_test_flat)
    logging.info("Isolation Forest training complete.")

    history = {"val_loss": [], "val_acc": []}
    client_data_split = None
    if federated:
        client_data_split = np.array_split(
            np.random.permutation(len(X_train)), num_clients
        )
        client_loaders = [
            DataLoader(
                TensorDataset(
                    torch.FloatTensor(X_train[idx]), torch.LongTensor(y_train[idx])
                ),
                batch_size=64,
                shuffle=True,
            )
            for idx in client_data_split
        ]

        global_model = copy.deepcopy(model).to(device)
        for round_num in range(rounds):
            local_weights = [
                FederatedClient(global_model, loader, num_classes, device).train(
                    1, 0.001
                )
                for loader in client_loaders
            ]
            global_model.load_state_dict(federated_average(local_weights))
            val_loss, val_acc = evaluate(
                global_model, X_test, y_test, num_classes, device
            )
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            logging.info(
                f"Round {round_num + 1}/{rounds} -> Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
        final_model = global_model
    else:  # Centralized
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=64,
            shuffle=True,
        )
        central_model = model.to(device)
        optimizer = optim.Adam(central_model.parameters(), lr=0.001)
        criterion = (
            nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
        )
        for epoch in range(rounds):
            central_model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = central_model(X_batch)
                loss = criterion(
                    outputs.squeeze(), y_batch.float() if num_classes == 2 else y_batch
                )
                loss.backward()
                optimizer.step()
            val_loss, val_acc = evaluate(
                central_model, X_test, y_test, num_classes, device
            )
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            logging.info(
                f"Epoch {epoch + 1}/{rounds} -> Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
        final_model = central_model

    _, _, y_pred, y_pred_proba = evaluate(
        final_model, X_test, y_test, num_classes, device, return_preds=True
    )
    labels = (
        ["Benign", "Attack"]
        if num_classes == 2
        else processed_data["processor"].encoders["attack_type"].classes_
    )
    labels = [label.capitalize() for label in labels]

    logging.info("Model training and evaluation complete.")
    logging.info(
        f"Final Classification Report ({detection_type.title()} - {'Federated' if federated else 'Centralized'}):\n"
        f"{classification_report(y_test, y_pred, target_names=labels)}"
    )

    results = {
        "history": history,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "labels": labels,
        "num_classes": num_classes,
        "df": df,
        "model": final_model,
        "y_train": y_train,
        "feature_names": feature_names,
        "X_test": X_test,
        "client_data_split": client_data_split,
        "anomaly_scores": anomaly_scores,
    }
    return results


def evaluate(model, X_test, y_test, num_classes, device, return_preds=False):
    model.eval()
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=256,
        shuffle=False,
    )
    criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
    val_loss, correct, total = 0, 0, 0
    all_preds, all_probs = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(
                outputs.squeeze(), y_batch.float() if num_classes == 2 else y_batch
            )
            val_loss += loss.item()

            probs = (
                torch.sigmoid(outputs.squeeze())
                if num_classes == 2
                else torch.softmax(outputs, dim=1)
            )
            preds = (
                (probs > 0.5).long() if num_classes == 2 else torch.argmax(probs, dim=1)
            )

            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            if return_preds:
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

    avg_loss = val_loss / len(test_loader)
    avg_acc = correct / total
    if return_preds:
        return avg_loss, avg_acc, np.array(all_preds), np.array(all_probs)
    return avg_loss, avg_acc


def generate_all_visualizations(results, mode, detection_type):
    title_suffix = f"({detection_type.title()} - {mode.title()})"
    plot_training_progress(results["history"], f"Training Progress {title_suffix}")
    plot_confusion_matrix(
        results["y_test"],
        results["y_pred"],
        results["labels"],
        f"Confusion Matrix {title_suffix}",
    )
    plot_roc_curve(
        results["y_test"],
        results["y_pred_proba"],
        results["labels"],
        results["num_classes"],
        f"ROC Curve {title_suffix}",
    )
    plot_pr_curve(
        results["y_test"],
        results["y_pred_proba"],
        results["labels"],
        results["num_classes"],
        f"Precision-Recall Curve {title_suffix}",
    )
    # plot_metrics_per_class(
    #     results["y_test"],
    #     results["y_pred"],
    #     results["labels"],
    #     f"Metrics Per Class {title_suffix}",
    # )
    # plot_prediction_probability_distribution(
    #     results["y_pred_proba"],
    #     results["num_classes"],
    #     f"Prediction Probabilities {title_suffix}",
    # )
    # plot_feature_importance(
    #     results["feature_names"], results["model"], f"Feature Importance {title_suffix}"
    # )
    plot_violin_probabilities(
        results["y_test"],
        results["y_pred_proba"],
        results["labels"],
        f"Prediction Certainty {title_suffix}",
    )
    plot_anomaly_scores(
        results["y_test"],
        results["anomaly_scores"],
        results["labels"],
        f"Anomaly Score Distribution {title_suffix}",
    )
    if mode == "Federated" and results["client_data_split"] is not None:
        plot_class_distribution_per_client(
            results["client_data_split"],
            results["y_train"],
            results["labels"],
            f"Class Distribution per Client {title_suffix}",
        )
filepath = "./evse_scenario_classification.csv"
# # --- Run All Analyses ---
# fed_binary_results = run_analysis(
#     filepath, detection_type="binary", federated=True, rounds=5
# )

# cent_binary_results = run_analysis(
#     filepath, detection_type="binary", federated=False, rounds=5
# )

# plot_sunburst_attacks(
#     fed_binary_results["df"], "Sunburst Hierarchy of Attack Types"
# )


fed_multi_results = run_analysis(
    filepath, detection_type="multiclass", federated=True, rounds=5
)
cent_multi_results = run_analysis(
    filepath, detection_type="multiclass", federated=False, rounds=1
)
# --- Generate Visualizations ---
logging.info("\n" + "=" * 30 + "\nGENERATING VISUALIZATIONS\n" + "=" * 30)

# General plots (run once using one of the results)
# plot_data_overview(fed_binary_results["df"], "Dataset Overview")
# plot_attack_distribution(
#     fed_binary_results["df"], "Overall Attack Type Distribution"
# )

# plot_feature_correlation_heatmap(
#     fed_binary_results["df"],
#     fed_binary_results["feature_names"][:25],
#     "Feature Correlation Heatmap (Top 25)",
# )
# plot_feature_distribution(
#     fed_binary_results["df"],
#     "bidirectional_duration_ms",
#     "Feature Distribution for bidirectional_duration_ms",
# )
# plot_parallel_coordinates(
#     fed_binary_results["df"],
#     fed_binary_results["feature_names"][:7],
#     "Parallel Coordinates of Key Features",
# )
# # Analysis-specific plots
# generate_all_visualizations(fed_binary_results, "Federated", "Binary")
# generate_all_visualizations(cent_binary_results, "Centralized", "Binary")

# Comparative plots
# plot_federated_vs_centralized(
#     fed_binary_results["history"],
#     cent_binary_results["history"],
#     "Binary Classification - Federated vs Centralized Accuracy",
# )
generate_all_visualizations(fed_multi_results, "Federated", "Multiclass")
generate_all_visualizations(cent_multi_results, "Centralized", "Multiclass")



plot_federated_vs_centralized(
    fed_multi_results["history"],
    cent_multi_results["history"],
    "Multiclass Classification - Federated vs Centralized Accuracy",
)
# PCA Plots
plot_pca_2d(
    fed_multi_results["X_test"],
    fed_multi_results["y_test"],
    fed_multi_results["labels"],
    "2D PCA Visualization of Test Set Features",
)
plot_pca_3d(
    fed_multi_results["X_test"],
    fed_multi_results["y_test"],
    fed_multi_results["labels"],
    "3D PCA Visualization of Test Set Features",
)

logging.info("\nAnalysis complete. All figures have been displayed and saved to the 'results' folder.")





