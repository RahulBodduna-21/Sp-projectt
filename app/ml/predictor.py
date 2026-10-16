import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from app.core.exceptions import MLError
from app.core.logging_config import logger

NON_FEATURE_COLS = [
    "session_id", "window_start", "window_end", "ground_truth_label",
    "latitude", "longitude", "speed_kmh", "gps_accuracy_m", "vehicle_type"
]

def generate_predictions(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    df_features: pd.DataFrame,
    feature_columns: List[str]
) -> pd.DataFrame:
    """Generates predictions and actual confidence scores for feature windows."""
    if df_features is None or df_features.empty:
        raise MLError("Cannot generate predictions on empty feature dataframe.")

    missing_cols = [c for c in feature_columns if c not in df_features.columns]
    if missing_cols:
        raise MLError(f"Missing required feature columns for model inference: {missing_cols}")

    X_pred = df_features[feature_columns].values
    X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)

    pred_indices = model.predict(X_pred)
    probas = model.predict_proba(X_pred)
    confidences = np.max(probas, axis=1)

    predicted_labels = label_encoder.inverse_transform(pred_indices)

    df_out = df_features.copy()
    df_out["predicted_class"] = predicted_labels
    df_out["confidence"] = confidences

    # Ensure required prediction schema columns
    required_schema = [
        "session_id", "window_start", "window_end", "ground_truth_label",
        "predicted_class", "confidence", "impact_score_mean", "latitude",
        "longitude", "speed_kmh", "gps_accuracy_m", "vehicle_type"
    ]
    
    # Rename impact_score_mean to impact_score for schema compatibility
    if "impact_score_mean" in df_out.columns:
        df_out["impact_score"] = df_out["impact_score_mean"]
    elif "impact_score" not in df_out.columns:
        df_out["impact_score"] = 0.0

    if "acc_mag_dominant_freq" in df_out.columns:
        df_out["dominant_frequency"] = df_out["acc_mag_dominant_freq"]
    elif "dominant_frequency" not in df_out.columns:
        df_out["dominant_frequency"] = 0.0

    output_cols = [
        "session_id", "window_start", "window_end", "ground_truth_label",
        "predicted_class", "confidence", "impact_score", "latitude",
        "longitude", "speed_kmh", "gps_accuracy_m", "dominant_frequency", "vehicle_type"
    ]
    
    # Fill missing optional columns
    for c in output_cols:
        if c not in df_out.columns:
            df_out[c] = np.nan

    logger.info(f"Generated predictions for {len(df_out)} windows. Mean confidence: {np.mean(confidences):.3f}")
    return df_out[output_cols]
