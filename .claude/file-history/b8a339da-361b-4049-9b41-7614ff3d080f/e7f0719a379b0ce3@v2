"""
Model architecture selection helpers: baselines, TCN, knowledge distillation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def xgboost_baseline(X, y, task_type="classification"):
    """Establish a strong baseline with XGBoost using 5-fold CV."""
    import xgboost as xgb

    if task_type == "classification":
        model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric="logloss",
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        )
    scoring = "accuracy" if task_type == "classification" else "neg_root_mean_squared_error"
    scores = cross_val_score(model, X, y, cv=5, scoring=scoring, n_jobs=-1)
    return {"mean_score": scores.mean(), "std_score": scores.std(), "model": model}


def get_model_recommendation(data_type, n_samples, task_type, constraints=None):
    """Recommend models based on data characteristics and constraints."""
    if constraints is None:
        constraints = {}
    rec = {"recommended_models": [], "notes": [], "reasoning": []}
    interpretability = constraints.get("interpretability", False)
    latency_ms = constraints.get("latency_ms", float("inf"))

    if data_type == "tabular":
        if n_samples < 1000:
            rec["recommended_models"] = ["LogisticRegression", "Ridge", "DecisionTree"]
            rec["reasoning"].append("Small dataset: prefer simple models")
        elif n_samples < 100000:
            rec["recommended_models"] = ["XGBoost", "LightGBM", "RandomForest"]
            rec["reasoning"].append("Medium dataset: tree-based ensembles excel")
        else:
            rec["recommended_models"] = ["XGBoost", "LightGBM", "TabNet"]
            rec["reasoning"].append("Large dataset: boosting + optional deep learning")
        if interpretability:
            rec["recommended_models"] = ["LogisticRegression", "EBM", "GAM"]
            rec["notes"].append("Interpretability required: glass-box models")
    elif data_type == "image":
        rec["recommended_models"] = ["ResNet", "EfficientNet", "ConvNeXt"]
        rec["reasoning"].append("Image data: transfer learning with CNNs")
    elif data_type == "text":
        rec["recommended_models"] = ["BERT", "RoBERTa", "DistilBERT"]
        rec["reasoning"].append("Text data: fine-tune pre-trained language models")
    elif data_type == "timeseries":
        rec["recommended_models"] = ["Prophet", "XGBoost with lags", "TCN", "LSTM"]
        rec["reasoning"].append("Time series: specialized temporal architectures")
    return rec


class TemporalBlock(nn.Module):
    """TCN temporal block with residual connections."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """Temporal Convolutional Network for sequence modeling."""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining hard and soft targets."""

    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, targets):
        hard_loss = F.cross_entropy(student_logits, targets)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


class RegularizedNet(nn.Module):
    """Neural network with dropout regularization."""

    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)


class BatchNormNet(nn.Module):
    """Neural network with batch normalization before activations."""

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = torch.relu(x)
        x = self.bn2(self.fc2(x))
        x = torch.relu(x)
        return self.fc3(x)
