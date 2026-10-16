import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from app.core.config import AppConfig, load_config
from app.core.logging_config import logger
from app.ml.trainer import train_classifier

def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_encoder: LabelEncoder,
    config: Optional[AppConfig] = None
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluates classifier using session-aware group splitting to prevent data leakage."""
    if config is None:
        config = load_config()

    unique_groups = np.unique(groups)
    num_sessions = len(unique_groups)

    # Check minimum class counts for stratified splitting
    _, class_counts = np.unique(y, return_counts=True)
    can_stratify = np.min(class_counts) >= 2

    if num_sessions > 1:
        logger.info(f"Using session-aware GroupShuffleSplit across {num_sessions} sessions to prevent data leakage...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=config.model.random_state)
        train_idx, test_idx = next(gss.split(X, y, groups))
        test_strategy = "GroupShuffleSplit (Session-Aware)"
    elif can_stratify and len(y) >= 4:
        logger.info("Single session detected. Using Stratified train_test_split (documented session restriction)...")
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=0.3, random_state=config.model.random_state, stratify=y
        )
        test_strategy = "Stratified train_test_split (Single Session Fallback)"
    else:
        logger.info("Single session with small sample count detected. Using random train_test_split...")
        test_size = 0.5 if len(y) <= 3 else 0.3
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=config.model.random_state
        )
        test_strategy = "Simple train_test_split (Small Dataset Fallback)"

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    clf = train_classifier(X_train, y_train, config=config)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

    # Overall metrics
    acc = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    p_class, r_class, f1_class, support_class = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    
    classes_in_test = np.unique(np.concatenate([y_test, y_pred]))
    per_class_report = {}
    for i, class_idx in enumerate(classes_in_test):
        c_name = str(label_encoder.classes_[class_idx])
        per_class_report[c_name] = {
            "precision": float(p_class[i]) if i < len(p_class) else 0.0,
            "recall": float(r_class[i]) if i < len(r_class) else 0.0,
            "f1_score": float(f1_class[i]) if i < len(f1_class) else 0.0,
            "support": int(support_class[i]) if i < len(support_class) else 0
        }

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(label_encoder.classes_)))
    try:
        clf_report_text = classification_report(
            y_test, y_pred,
            labels=classes_in_test,
            target_names=[str(label_encoder.classes_[i]) for i in classes_in_test],
            zero_division=0
        )
    except Exception:
        clf_report_text = f"Accuracy: {acc:.4f}, Macro F1: {f1_macro:.4f}"

    metrics = {
        "test_strategy": test_strategy,
        "num_train_samples": int(len(X_train)),
        "num_test_samples": int(len(X_test)),
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "per_class_metrics": per_class_report,
        "classification_report_text": clf_report_text,
        "confusion_matrix": cm.tolist(),
        "classes": list(label_encoder.classes_)
    }

    logger.info(f"Evaluation finished. Accuracy: {acc:.4f}, Macro F1: {f1_macro:.4f}")
    return metrics, y_test, y_pred
