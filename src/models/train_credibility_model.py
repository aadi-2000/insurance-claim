"""
Credibility Model — Synthetic Data Generation + Random Forest Training

Generates realistic synthetic insurance claim data with 11 features,
trains a Random Forest classifier, evaluates it, and saves the artifact.

Usage:
    python -m src.models.train_credibility_model          # from project root
    python src/models/train_credibility_model.py           # direct

Output:
    data/models/credibility_rf_model.joblib      — trained model
    data/models/credibility_model_metadata.json  — feature names, metrics, params
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

# ============================================================
# Constants
# ============================================================

FEATURE_NAMES = [
    "claim_approval_rate",
    "claim_frequency",
    "flag_rate",
    "no_rejections",
    "policy_tenure_norm",
    "docs_complete",
    "image_integrity",
    "amount_reasonableness",
    "proc_diag_consistency",
    "network_hospital",
    "timing_score",
]

LABEL_MAP = {
    0: "VERY_LOW",   # Fraudulent / highly suspicious
    1: "LOW",        # Concerning — needs review
    2: "MODERATE",   # Acceptable but not strong
    3: "HIGH",       # Trustworthy claimant
}

N_SAMPLES = 10000
RANDOM_STATE = 42

# Resolve project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models"


# ============================================================
# Synthetic Data Generator
# ============================================================

def generate_synthetic_data(n_samples: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate realistic synthetic insurance claim data.

    The generator creates 4 claimant profiles with distinct feature
    distributions, simulating real-world claim patterns:

    - HIGH credibility (40%): Long-tenure, clean history, network hospitals
    - MODERATE credibility (25%): Mixed signals, some minor flags
    - LOW credibility (20%): New policies, non-network, some inconsistencies
    - VERY_LOW credibility (15%): Flagged history, manipulated docs, mismatches
    """
    rng = np.random.default_rng(seed)
    records = []

    profile_distribution = [
        (3, 0.40),  # HIGH
        (2, 0.25),  # MODERATE
        (1, 0.20),  # LOW
        (0, 0.15),  # VERY_LOW
    ]

    for label, fraction in profile_distribution:
        count = int(n_samples * fraction)
        for _ in range(count):
            records.append(_generate_sample(rng, label))

    df = pd.DataFrame(records)
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def _generate_sample(rng: np.random.Generator, label: int) -> dict:
    """Generate a single sample based on the credibility label."""

    if label == 3:  # HIGH credibility
        return {
            "claim_approval_rate": _clip(rng.beta(8, 2)),
            "claim_frequency": _clip(rng.beta(2, 5)),
            "flag_rate": _clip(rng.beta(1, 20)),
            "no_rejections": _clip(rng.beta(9, 1)),
            "policy_tenure_norm": _clip(rng.beta(5, 2)),
            "docs_complete": rng.choice([1.0, 1.0, 1.0, 0.3], p=[0.90, 0.05, 0.03, 0.02]),
            "image_integrity": _clip(rng.normal(0.92, 0.05)),
            "amount_reasonableness": rng.choice([1.0, 0.85, 0.6, 0.4], p=[0.45, 0.35, 0.15, 0.05]),
            "proc_diag_consistency": rng.choice([1.0, 0.5, 0.3], p=[0.85, 0.10, 0.05]),
            "network_hospital": rng.choice([1.0, 0.5], p=[0.80, 0.20]),
            "timing_score": rng.choice([1.0, 0.7, 0.3], p=[0.85, 0.12, 0.03]),
            "label": label,
        }

    elif label == 2:  # MODERATE credibility
        return {
            "claim_approval_rate": _clip(rng.beta(5, 3)),
            "claim_frequency": _clip(rng.beta(3, 4)),
            "flag_rate": _clip(rng.beta(1.5, 10)),
            "no_rejections": _clip(rng.beta(6, 2)),
            "policy_tenure_norm": _clip(rng.beta(3, 3)),
            "docs_complete": rng.choice([1.0, 1.0, 0.3], p=[0.70, 0.15, 0.15]),
            "image_integrity": _clip(rng.normal(0.82, 0.10)),
            "amount_reasonableness": rng.choice([1.0, 0.85, 0.6, 0.4], p=[0.25, 0.35, 0.25, 0.15]),
            "proc_diag_consistency": rng.choice([1.0, 0.5, 0.3], p=[0.60, 0.25, 0.15]),
            "network_hospital": rng.choice([1.0, 0.5], p=[0.55, 0.45]),
            "timing_score": rng.choice([1.0, 0.7, 0.3], p=[0.60, 0.25, 0.15]),
            "label": label,
        }

    elif label == 1:  # LOW credibility
        return {
            "claim_approval_rate": _clip(rng.beta(3, 5)),
            "claim_frequency": _clip(rng.beta(4, 3)),
            "flag_rate": _clip(rng.beta(3, 6)),
            "no_rejections": _clip(rng.beta(3, 4)),
            "policy_tenure_norm": _clip(rng.beta(2, 5)),
            "docs_complete": rng.choice([1.0, 0.3], p=[0.45, 0.55]),
            "image_integrity": _clip(rng.normal(0.70, 0.12)),
            "amount_reasonableness": rng.choice([1.0, 0.85, 0.6, 0.4], p=[0.10, 0.20, 0.35, 0.35]),
            "proc_diag_consistency": rng.choice([1.0, 0.5, 0.3], p=[0.30, 0.35, 0.35]),
            "network_hospital": rng.choice([1.0, 0.5], p=[0.30, 0.70]),
            "timing_score": rng.choice([1.0, 0.7, 0.3], p=[0.30, 0.35, 0.35]),
            "label": label,
        }

    else:  # label == 0, VERY_LOW credibility
        return {
            "claim_approval_rate": _clip(rng.beta(2, 7)),
            "claim_frequency": _clip(rng.beta(5, 2)),
            "flag_rate": _clip(rng.beta(5, 3)),
            "no_rejections": _clip(rng.beta(2, 6)),
            "policy_tenure_norm": _clip(rng.beta(1.5, 6)),
            "docs_complete": rng.choice([1.0, 0.3], p=[0.25, 0.75]),
            "image_integrity": _clip(rng.normal(0.55, 0.15)),
            "amount_reasonableness": rng.choice([1.0, 0.85, 0.6, 0.4], p=[0.05, 0.10, 0.25, 0.60]),
            "proc_diag_consistency": rng.choice([1.0, 0.5, 0.3], p=[0.15, 0.25, 0.60]),
            "network_hospital": rng.choice([1.0, 0.5], p=[0.20, 0.80]),
            "timing_score": rng.choice([1.0, 0.7, 0.3], p=[0.15, 0.25, 0.60]),
            "label": label,
        }


def _clip(value: float) -> float:
    """Clip value to [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


# ============================================================
# Model Training
# ============================================================

def train_model(df: pd.DataFrame) -> dict:
    """
    Train a Random Forest classifier with hyperparameter tuning.

    Returns dict with: model, metrics, best_params, feature_importances
    """
    X = df[FEATURE_NAMES].values
    y = df["label"].values

    # Split: 80% train, 20% test (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set:     {X_test.shape[0]} samples")
    print(f"Features:     {X_train.shape[1]}")
    print(f"Classes:      {np.unique(y_train)}")
    print()

    # Hyperparameter grid search
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample"],
    }

    print("Running GridSearchCV (5-fold)...")
    base_rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(
        base_rf, param_grid,
        cv=5, scoring="accuracy",
        n_jobs=-1, verbose=0,
        refit=True,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best params: {best_params}")
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
    print()

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    # Multi-class AUC (one-vs-rest)
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except ValueError:
        auc = 0.0

    print("=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC (OVR): {auc:.4f}")
    print()

    print("Classification Report:")
    target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, index=target_names, columns=target_names))
    print()

    # Feature importances
    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print("Feature Importances:")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:30s} {row['importance']:.4f} {bar}")
    print()

    # Cross-validation score on full dataset
    cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="accuracy")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "auc_ovr": round(auc, 4),
        "cv_accuracy_mean": round(cv_scores.mean(), 4),
        "cv_accuracy_std": round(cv_scores.std(), 4),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
    }

    return {
        "model": best_model,
        "metrics": metrics,
        "best_params": best_params,
        "feature_importances": dict(zip(FEATURE_NAMES, [round(x, 4) for x in importances])),
        "confusion_matrix": cm.tolist(),
    }


# ============================================================
# Save Artifacts
# ============================================================

def save_artifacts(result: dict, model_dir: Path) -> None:
    """Save the trained model and metadata."""
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = model_dir / "credibility_rf_model.joblib"
    joblib.dump(result["model"], model_path)
    print(f"Model saved: {model_path}")

    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "feature_names": FEATURE_NAMES,
        "label_map": LABEL_MAP,
        "n_features": len(FEATURE_NAMES),
        "n_classes": len(LABEL_MAP),
        "best_params": result["best_params"],
        "metrics": result["metrics"],
        "feature_importances": result["feature_importances"],
        "confusion_matrix": result["confusion_matrix"],
        "training_samples": N_SAMPLES,
        "random_state": RANDOM_STATE,
    }
    metadata_path = model_dir / "credibility_model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("CREDIBILITY MODEL TRAINING PIPELINE")
    print("=" * 60)
    print()

    # Step 1: Generate data
    print("[1/3] Generating synthetic training data...")
    df = generate_synthetic_data(N_SAMPLES)
    print(f"Generated {len(df)} samples")
    print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")
    print()

    # Step 2: Train model
    print("[2/3] Training Random Forest model...")
    result = train_model(df)
    print()

    # Step 3: Save
    print("[3/3] Saving artifacts...")
    save_artifacts(result, MODEL_DIR)
    print()
    print("Done!")


if __name__ == "__main__":
    main()
