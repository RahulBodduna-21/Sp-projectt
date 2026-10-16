import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from app.core.exceptions import DatasetValidationError
from app.core.logging_config import logger

REQUIRED_COLUMNS = [
    "session_id", "id", "timestamp", "elapsed_time_s",
    "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
    "latitude", "longitude", "speed_kmh", "gps_accuracy_m", "label"
]

def normalize_vehicle_type(v_type: Any) -> str:
    """Normalizes vehicle type strings to standard 2-WHEELER, 3-WHEELER, 4-WHEELER."""
    if v_type is None or pd.isna(v_type):
        return "2-WHEELER"
    clean = str(v_type).strip().upper().replace(" ", "").replace("_", "").replace("-", "")
    if "2W" in clean or "BIKE" in clean or "MOTORCYCLE" in clean or "SCOOTER" in clean or "2WHEELER" in clean:
        return "2-WHEELER"
    if "3W" in clean or "AUTO" in clean or "RICKSHAW" in clean or "3WHEELER" in clean:
        return "3-WHEELER"
    if "4W" in clean or "CAR" in clean or "FOUR" in clean or "4WHEELER" in clean:
        return "4-WHEELER"
    return str(v_type).strip().upper()

def normalize_label(label: Any) -> Optional[str]:
    if label is None or pd.isna(label):
        return None
    cleaned = " ".join(str(label).strip().upper().replace("_", " ").replace("-", " ").split())
    if cleaned in ["POTHOLE", "POT HOLE"]:
        return "POTHOLE"
    if cleaned in ["SPEED BREAKER", "SPEEDBREAKER", "SPEED BUMP"]:
        return "SPEED BREAKER"
    if cleaned in ["SMOOTH", "SMOOTH ROAD"]:
        return "SMOOTH"
    if cleaned in ["ROUGH", "ROUGH ROAD"]:
        return "ROUGH"
    return cleaned

def validate_dataset(df: pd.DataFrame, default_vehicle_type: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Strictly validates uploaded CSV dataframe and generates a comprehensive dataset quality report."""
    if df is None or df.empty:
        raise DatasetValidationError("Dataset is empty or None.")
    
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DatasetValidationError(f"Missing required columns: {missing_cols}")
    
    df_clean = df.copy()
    
    # Assign or normalize vehicle_type metadata
    if "vehicle_type" in df_clean.columns and df_clean["vehicle_type"].notna().any():
        df_clean["vehicle_type"] = df_clean["vehicle_type"].apply(normalize_vehicle_type)
    elif default_vehicle_type:
        df_clean["vehicle_type"] = normalize_vehicle_type(default_vehicle_type)
    else:
        df_clean["vehicle_type"] = "2-WHEELER"
    
    # Parse timestamps
    try:
        df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce", utc=True)
    except Exception as e:
        raise DatasetValidationError(f"Failed to parse timestamps: {e}")
    
    invalid_time_mask = df_clean["timestamp"].isna()
    
    # Label normalization
    df_clean["label"] = df_clean["label"].apply(normalize_label)
    
    # Numeric conversions
    numeric_cols = [
        "elapsed_time_s", "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z", "latitude", "longitude",
        "speed_kmh", "gps_accuracy_m"
    ]
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
    
    # Sensor validity masks
    valid_acc_mask = df_clean["acc_x"].notna() & df_clean["acc_y"].notna() & df_clean["acc_z"].notna()
    valid_gyro_mask = df_clean["gyro_x"].notna() & df_clean["gyro_y"].notna() & df_clean["gyro_z"].notna()
    valid_gps_mask = (
        df_clean["latitude"].notna() &
        df_clean["longitude"].notna() &
        (df_clean["latitude"] >= -90.0) & (df_clean["latitude"] <= 90.0) &
        (df_clean["longitude"] >= -180.0) & (df_clean["longitude"] <= 180.0) &
        (df_clean["latitude"] != 0.0) & (df_clean["longitude"] != 0.0)
    )
    
    # Ensure chronological order per session_id
    df_clean = df_clean.sort_values(by=["session_id", "timestamp"]).reset_index(drop=True)
    
    total_rows = len(df)
    missing_counts = df_clean.isna().sum().to_dict()
    num_sessions = df_clean["session_id"].nunique()
    vehicle_type_val = str(df_clean["vehicle_type"].iloc[0])
    
    # Calculate duration and estimated sampling rate
    durations = []
    sampling_rates = []
    for sid, group in df_clean.groupby("session_id"):
        if len(group) > 1 and group["timestamp"].notna().sum() > 1:
            dt = group["timestamp"].diff().dt.total_seconds()
            dt_valid = dt[dt > 0]
            if len(dt_valid) > 0:
                med_dt = dt_valid.median()
                if med_dt > 0:
                    sampling_rates.append(1.0 / med_dt)
            dur = (group["timestamp"].max() - group["timestamp"].min()).total_seconds()
            durations.append(dur)
    
    est_sr = float(np.median(sampling_rates)) if sampling_rates else 50.0
    total_duration = sum(durations) if durations else 0.0
    
    label_dist = df_clean["label"].value_counts(dropna=False).to_dict()
    label_dist_clean = {str(k): int(v) for k, v in label_dist.items()}
    
    valid_rows = len(df_clean[~invalid_time_mask])
    invalid_rows = total_rows - valid_rows
    
    pothole_labels = int(label_dist_clean.get("POTHOLE", 0))
    speed_breaker_labels = int(label_dist_clean.get("SPEED BREAKER", 0))
    rough_labels = int(label_dist_clean.get("ROUGH", 0))
    smooth_labels = int(label_dist_clean.get("SMOOTH", 0))
    
    report = {
        "vehicle_type": vehicle_type_val,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "missing_values": missing_counts,
        "number_of_sessions": int(num_sessions),
        "total_duration_seconds": float(round(total_duration, 1)),
        "estimated_sampling_rate_hz": float(round(est_sr, 2)),
        "label_distribution": label_dist_clean,
        "pothole_labels": pothole_labels,
        "speed_breaker_labels": speed_breaker_labels,
        "rough_labels": rough_labels,
        "smooth_labels": smooth_labels,
        "accelerometer_valid_rows": int(valid_acc_mask.sum()),
        "gyroscope_valid_rows": int(valid_gyro_mask.sum()),
        "gps_coverage": {
            "valid_gps_count": int(valid_gps_mask.sum()),
            "coverage_percentage": float(round(valid_gps_mask.mean() * 100, 2)) if total_rows > 0 else 0.0
        }
    }
    
    logger.info(f"Dataset validation ({vehicle_type_val}) completed. Total rows: {total_rows}, Sessions: {num_sessions}, Est SR: {est_sr:.1f} Hz")
    return df_clean, report
