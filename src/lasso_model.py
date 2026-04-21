import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, accuracy_score


class LassoModel:
    """L1-regularised multinomial logistic regression for bubble state prediction.

    Uses LogisticRegressionCV so the regularisation strength C is selected by
    cross-validation on the training fold (no separate tuning set required).
    SMOTE is applied upstream by the caller before passing X_train / y_train.
    """

    def __init__(self, cv=5, max_iter=10000, random_state=42):
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.feature_names_ = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            feature_names=None, **kwargs):
        self.feature_names_ = feature_names
        self.model = LogisticRegressionCV(
            penalty="l1",
            solver="saga",
            multi_class="ovr", # fixes convergence issues with small datasets and many classes
            class_weight="balanced",
            cv=self.cv,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return _compute_metrics(y, y_pred)

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        coef = np.abs(self.model.coef_)        # (n_classes, n_features)
        importance = coef.mean(axis=0)
        idx = self.feature_names_ if self.feature_names_ is not None else range(len(importance))
        return pd.Series(importance, index=idx).sort_values(ascending=False)


def _compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":       accuracy_score(y_true, y_pred),
        "macro_f1":       f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "weighted_f1":    f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_not_bubble":  f1_score(y_true, y_pred, labels=[0], average="micro", zero_division=0),
        "f1_creation":    f1_score(y_true, y_pred, labels=[1], average="micro", zero_division=0),
        "f1_collapse":    f1_score(y_true, y_pred, labels=[2], average="micro", zero_division=0),
    }
