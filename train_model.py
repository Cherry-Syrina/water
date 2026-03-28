"""
train_model.py
--------------
Trains a Gradient Boosting Classifier on the water quality dataset.
Handles preprocessing, splitting, training, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_data import generate_dataset, get_feature_names


# ── Hyperparameters ────────────────────────────────────────────────────────────
MODEL_PARAMS = {
    "n_estimators":   150,
    "max_depth":      5,
    "learning_rate":  0.1,
    "subsample":      0.85,
    "random_state":   42,
}

TEST_SIZE    = 0.20
RANDOM_STATE = 42
# ───────────────────────────────────────────────────────────────────────────────


def preprocess(df: pd.DataFrame):
    """
    Split dataset into train/test sets and apply StandardScaler.

    Returns
    -------
    X_train_s, X_test_s : scaled feature arrays
    y_train, y_test     : label arrays
    scaler              : fitted StandardScaler (needed for inference)
    """
    feature_cols = get_feature_names()
    X = df[feature_cols].values
    y = df["Potability"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_test_s, y_train, y_test, scaler


def build_model() -> GradientBoostingClassifier:
    """Instantiate model with tuned hyperparameters."""
    return GradientBoostingClassifier(**MODEL_PARAMS)


def evaluate(model, X_test_s, y_test) -> dict:
    """
    Compute evaluation metrics on held-out test set.

    Returns
    -------
    dict with accuracy, roc_auc, confusion_matrix, classification_report
    """
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    metrics = {
        "accuracy":              accuracy_score(y_test, y_pred),
        "roc_auc":               roc_auc_score(y_test, y_prob),
        "confusion_matrix":      confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
                                     y_test, y_pred,
                                     target_names=["Non-Potable", "Potable"],
                                     output_dict=True,
                                 ),
        "y_pred":                y_pred,
    }
    return metrics


def train_pipeline(n_samples: int = 3000):
    """
    End-to-end training pipeline.

    Returns
    -------
    model   : trained GradientBoostingClassifier
    scaler  : fitted StandardScaler
    metrics : evaluation dict
    X_test_s, y_test : held-out data for downstream plotting
    """
    df = generate_dataset(n_samples=n_samples)
    X_train_s, X_test_s, y_train, y_test, scaler = preprocess(df)

    model = build_model()
    model.fit(X_train_s, y_train)

    metrics = evaluate(model, X_test_s, y_test)
    return model, scaler, metrics, X_test_s, y_test


if __name__ == "__main__":
    print("Training Water Quality Prediction Model...")
    model, scaler, metrics, X_test_s, y_test = train_pipeline()

    print(f"\nAccuracy : {metrics['accuracy']*100:.2f}%")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
    print(report_df.round(3))
    print("\nModel training complete.")
