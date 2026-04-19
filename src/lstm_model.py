"""LSTM model for multi-class bubble state prediction.

SMOTE is not applied to sequence data (synthesising arbitrary 3-D windows is
ill-defined).  Instead class imbalance is addressed via inverse-frequency
weights in the CrossEntropyLoss, matching the approach in lstm.ipynb.
"""

from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score


def set_seed(seed: int = 42) -> None:
    """Replicate lstm.ipynb set_seed(): seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Core architecture
# ---------------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.3,
                 dense_size: int = 16, num_classes: int = 3):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, dense_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])   # last time-step
        return self.fc2(self.relu(self.fc1(out)))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Sequence creation (no coin-boundary crossing)
# ---------------------------------------------------------------------------

def create_sequences(X: np.ndarray, y: np.ndarray,
                     lookback: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window sequences from a contiguous block of data."""
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback: i])
        ys.append(y[i])
    if not Xs:
        return np.empty((0, lookback, X.shape[1])), np.empty(0, dtype=int)
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


def create_sequences_multi_coin(X_blocks: list[np.ndarray],
                                y_blocks: list[np.ndarray],
                                lookback: int = 14
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences from multiple coin blocks without crossing boundaries."""
    all_X, all_y = [], []
    for X_block, y_block in zip(X_blocks, y_blocks):
        Xs, ys = create_sequences(X_block, y_block, lookback)
        all_X.append(Xs)
        all_y.append(ys)
    if not all_X or all(len(a) == 0 for a in all_X):
        n_feat = X_blocks[0].shape[1] if X_blocks else 1
        return np.empty((0, lookback, n_feat)), np.empty(0, dtype=int)
    return np.concatenate(all_X), np.concatenate(all_y)


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------

class LSTMModel:
    """Scikit-learn–style wrapper around LSTMClassifier.

    Parameters
    ----------
    input_size : int
        Number of features per time step.
    class_weights : array-like of shape (3,) or None
        Manual per-class weights for CrossEntropyLoss.  If None, inverse
        frequency weights are computed from the training labels.
    """

    def __init__(self, input_size: int, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.3,
                 dense_size: int = 16, lookback: int = 14,
                 epochs: int = 15, batch_size: int = 32,
                 lr: float = 5e-4, weight_decay: float = 1e-4,
                 patience: int = 4, class_weights=None,
                 device: str = None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dense_size = dense_size
        self.lookback = lookback
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.class_weights = class_weights
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: LSTMClassifier | None = None
        self._best_state: dict | None = None

    # ------------------------------------------------------------------
    # Fitting (single-coin or multi-coin via pre-concatenated sequences)
    # ------------------------------------------------------------------

    def fit(self, X_seq: np.ndarray, y_seq: np.ndarray,
            X_val_seq: np.ndarray = None, y_val_seq: np.ndarray = None,
            **kwargs):
        """Fit on pre-built sequence arrays (shape: n, lookback, features)."""
        if len(X_seq) == 0:
            return self

        cw = self._build_class_weights(y_seq)
        criterion = nn.CrossEntropyLoss(weight=cw)

        loader = DataLoader(SequenceDataset(X_seq, y_seq),
                            batch_size=self.batch_size, shuffle=True)

        self.model = LSTMClassifier(
            self.input_size, self.hidden_size, self.num_layers,
            self.dropout, self.dense_size,
        ).to(self.device)

        opt = torch.optim.Adam(self.model.parameters(),
                               lr=self.lr, weight_decay=self.weight_decay)

        best_f1, patience_count = -1.0, 0
        for _ in range(self.epochs):
            self.model.train()
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                criterion(self.model(Xb), yb).backward()
                opt.step()

            if X_val_seq is not None and len(X_val_seq) > 0:
                val_f1 = self._eval_f1(X_val_seq, y_val_seq)
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    self._best_state = {k: v.clone()
                                        for k, v in self.model.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1
                    if patience_count >= self.patience:
                        break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        if self.model is None or len(X_seq) == 0:
            return np.array([], dtype=int)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_seq, dtype=torch.float32).to(self.device))
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        if self.model is None or len(X_seq) == 0:
            return np.empty((0, 3))
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(X_seq, dtype=torch.float32).to(self.device))
        return torch.softmax(logits, dim=1).cpu().numpy()

    def evaluate(self, X_seq: np.ndarray, y_seq: np.ndarray) -> dict:
        y_pred = self.predict(X_seq)
        if len(y_pred) == 0:
            return {k: float("nan") for k in
                    ["accuracy", "macro_f1", "weighted_f1",
                     "f1_not_bubble", "f1_creation", "f1_collapse"]}
        return _compute_metrics(y_seq, y_pred)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_class_weights(self, y: np.ndarray) -> torch.Tensor:
        if self.class_weights is not None:
            w = np.array(self.class_weights, dtype=np.float32)
        else:
            counts = np.bincount(y, minlength=3).astype(np.float32)
            w = 1.0 / (counts + 1e-8)
            w = w / w.sum() * 3          # normalise so sum == n_classes
        return torch.tensor(w, dtype=torch.float32).to(self.device)

    def _eval_f1(self, X_seq: np.ndarray, y_seq: np.ndarray) -> float:
        y_pred = self.predict(X_seq)
        if len(y_pred) == 0:
            return 0.0
        return f1_score(y_seq, y_pred, average="macro", zero_division=0)


def _compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":       accuracy_score(y_true, y_pred),
        "macro_f1":       f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "weighted_f1":    f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_not_bubble":  f1_score(y_true, y_pred, labels=[0], average="micro", zero_division=0),
        "f1_creation":    f1_score(y_true, y_pred, labels=[1], average="micro", zero_division=0),
        "f1_collapse":    f1_score(y_true, y_pred, labels=[2], average="micro", zero_division=0),
    }
