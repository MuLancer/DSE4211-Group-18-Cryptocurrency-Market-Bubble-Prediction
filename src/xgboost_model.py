import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)


_DEFAULT_PARAMS = dict(
    objective="multi:softmax",
    num_class=3,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

# Continuous distributions matching xgboost.ipynb hyperparameter_tuning()
_PARAM_DISTRIBUTIONS = dict(
    n_estimators     = randint(100, 600),
    max_depth        = randint(3, 10),
    learning_rate    = uniform(0.01, 0.19),   # loc + scale: range [0.01, 0.20]
    subsample        = uniform(0.6, 0.4),
    colsample_bytree = uniform(0.5, 0.5),
    min_child_weight = randint(1, 15),
    gamma            = uniform(0, 0.5),
    reg_alpha        = uniform(0, 1.0),
    reg_lambda       = uniform(0.5, 2.0),
)


class XGBoostModel:
    """XGBoost multi-class gradient boosting wrapper for bubble prediction.

    Optionally tunes hyper-parameters with RandomizedSearchCV on the training
    fold (using the validation set as a hold-out for early stopping).
    SMOTE is applied upstream by the caller.
    """

    def __init__(self, params: dict = None, tune: bool = False,
                 n_iter: int = 40, early_stopping_rounds: int = 30,
                 random_state: int = 42):
        self.params = {**_DEFAULT_PARAMS, **(params or {})}
        self.tune = tune
        self.n_iter = n_iter
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs):
        # Remap labels to 0-indexed — XGBoost requires labels in [0, num_class)
        # Some rolling windows lack certain classes (e.g. only [1,2] present).
        self._orig_classes = np.unique(y_train)
        y_mapped = self._remap_labels(y_train)

        if self.tune:
            self._tune(X_train, y_mapped)

        params = {**self.params, "num_class": len(self._orig_classes)}
        self.model = xgb.XGBClassifier(**params)

        # Mirror xgboost.ipynb train_model(): carve 10% from training for
        # early stopping so the test set is never seen during fitting.
        try:
            X_tr, X_es, y_tr, y_es = train_test_split(
                X_train, y_mapped,
                test_size=0.1,
                stratify=y_mapped,
                random_state=self.random_state,
            )
        except ValueError:
            X_tr, X_es, y_tr, y_es = train_test_split(
                X_train, y_mapped,
                test_size=0.1,
                random_state=self.random_state,
            )
        self.model.set_params(early_stopping_rounds=self.early_stopping_rounds)
        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_es, y_es)],
            verbose=False,
        )
        return self

    def _remap_labels(self, y: np.ndarray) -> np.ndarray:
        label_map = {c: i for i, c in enumerate(self._orig_classes)}
        return np.array([label_map[c] for c in y], dtype=int)

    def _inverse_remap(self, y: np.ndarray) -> np.ndarray:
        return np.array([self._orig_classes[i] for i in y], dtype=int)

    def _tune(self, X: np.ndarray, y: np.ndarray):
        """RandomizedSearchCV matching xgboost.ipynb hyperparameter_tuning()."""
        base_params = {
            "objective": "multi:softmax",
            "num_class": len(self._orig_classes),
            "eval_metric": "mlogloss",
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        base = xgb.XGBClassifier(**base_params)
        # shuffle=True + random_state matches the original notebook
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        search = RandomizedSearchCV(
            base, _PARAM_DISTRIBUTIONS,
            n_iter=self.n_iter,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            random_state=self.random_state,
            refit=True,
            verbose=0,
        )
        search.fit(X, y)
        self.params.update(search.best_estimator_.get_params())

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.asarray(self.model.predict(X))
        if preds.ndim > 1:
            preds = preds.argmax(axis=-1)
        preds = preds.ravel().astype(int)
        if hasattr(self, "_orig_classes"):
            preds = self._inverse_remap(preds)
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return _compute_metrics(y, y_pred)

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        scores = self.model.get_booster().get_score(importance_type="gain")
        return pd.Series(scores).sort_values(ascending=False)


def _compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    return {
        "accuracy":       accuracy_score(y_true, y_pred),
        "macro_f1":       f1_score(y_true, y_pred, average="macro",    zero_division=0),
        "weighted_f1":    f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_not_bubble":  f1_score(y_true, y_pred, labels=[0], average="micro", zero_division=0),
        "f1_creation":    f1_score(y_true, y_pred, labels=[1], average="micro", zero_division=0),
        "f1_collapse":    f1_score(y_true, y_pred, labels=[2], average="micro", zero_division=0),
    }
