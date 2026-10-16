import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import LabelEncoder
from app.core.exceptions import MLError
from app.core.logging_config import logger

NON_FEATURE_COLS = [
    "session_id", "window_start", "window_end", "ground_truth_label",
    "latitude", "longitude", "speed_kmh", "gps_accuracy_m", "vehicle_type"
]

def build_ml_dataset(
    df_features: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    label_encoder: Optional[LabelEncoder] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[LabelEncoder], List[str], pd.DataFrame]:
    """Prepares feature matrix X, target vector y, groups (session_id), label encoder, and feature names."""
    if df_features is None or df_features.empty:
        raise MLError("Input feature dataframe is empty.")

    # Determine feature columns
    if feature_columns is None:
        feature_columns = [c for c in df_features.columns if c not in NON_FEATURE_COLS]
    else:
        feature_columns = [c for c in feature_columns if c in df_features.columns]

    if not feature_columns:
        raise MLError("No feature columns available.")

    # Filter rows with valid ground_truth_label for training
    df_valid = df_features[df_features["ground_truth_label"].notna() & (df_features["ground_truth_label"] != "OTHER")].copy()
    
    if len(df_valid) == 0:
        logger.info("Unlabeled/Field dataset detected. Preparing feature matrix X for inference with pre-trained model...")
        X = df_features[feature_columns].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        groups = df_features["session_id"].values
        return X, None, groups, label_encoder, feature_columns, df_features

    X = df_valid[feature_columns].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Encode labels
    raw_labels = df_valid["ground_truth_label"].values
    if label_encoder is None:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(raw_labels)
    else:
        y = np.array([label_encoder.transform([lbl])[0] if lbl in label_encoder.classes_ else -1 for lbl in raw_labels])

    groups = df_valid["session_id"].values
    unique_classes = np.unique(y[y >= 0])
    
    if len(unique_classes) < 2:
        logger.info(f"Single target class ({unique_classes}) found in dataset. Preparing matrix for inference...")
        X_full = df_features[feature_columns].values
        X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
        return X_full, None, df_features["session_id"].values, label_encoder, feature_columns, df_features

    logger.info(f"Built ML dataset: {X.shape[0]} labeled samples, {X.shape[1]} features, {len(unique_classes)} classes, {len(np.unique(groups))} sessions.")
    return X, y, groups, label_encoder, feature_columns, df_valid
