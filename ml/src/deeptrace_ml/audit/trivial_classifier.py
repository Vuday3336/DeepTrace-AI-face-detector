"""The "trivial classifier" test: can simple features alone separate real from fake?

ROC-AUC 0.5 = no information, 1.0 = perfect separation. We report two models: logistic regression
(linear rules) and gradient boosting (non-linear rules like "width == 1024 and PNG"). Permutation
importance names WHICH features carry the shortcut.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrivialClassifierResult:
    feature_set: str
    n_samples: int
    n_features_used: int
    constant_features_dropped: list[str]
    auc: dict[str, dict[str, float]] = field(default_factory=dict)  # model -> {mean, std}
    top_features: list[dict[str, float]] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _models(seed: int) -> dict[str, Any]:
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
    }


def run_trivial_classifier(
    features: pd.DataFrame, labels: pd.Series, feature_set: str, seed: int, n_splits: int = 5
) -> TrivialClassifierResult:
    if len(features) != len(labels):
        raise ValueError("features and labels must have the same length")
    y = (labels.to_numpy() == "fake").astype(int)
    class_counts = np.bincount(y, minlength=2)
    if class_counts.min() < n_splits:
        raise ValueError(f"Need >= {n_splits} samples per class, got {class_counts.tolist()}")

    X = features.reset_index(drop=True).astype(float)
    valid_rows = X.notna().all(axis=1).to_numpy()
    X, y = X[valid_rows], y[valid_rows]
    constant = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=constant)

    result = TrivialClassifierResult(
        feature_set=feature_set,
        n_samples=int(len(X)),
        n_features_used=int(X.shape[1]),
        constant_features_dropped=constant,
    )
    if X.shape[1] == 0:
        result.auc = {name: {"mean": 0.5, "std": 0.0} for name in _models(seed)}
        result.note = "All features are constant: no information about the label (ideal)."
        return result

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for name, model in _models(seed).items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        result.auc[name] = {"mean": float(scores.mean()), "std": float(scores.std())}

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    booster = HistGradientBoostingClassifier(random_state=seed).fit(X_tr, y_tr)
    importance = permutation_importance(booster, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=seed)
    order = np.argsort(importance.importances_mean)[::-1][:5]
    result.top_features = [
        {"feature": str(X.columns[i]), "auc_drop": float(importance.importances_mean[i])} for i in order
    ]
    return result
