import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from app.core.config import AppConfig, load_config
from app.core.logging_config import logger
from app.features.time_domain import compute_time_domain_features_1d
from app.features.frequency_domain import compute_fft_features_1d

def extract_features_from_window(
    window_df: pd.DataFrame,
    sampling_rate_hz: float = 50.0,
    config_stage: str = "E"
) -> Dict[str, Any]:
    """Extracts statistical time-domain and FFT features from a single window dataframe."""
    row_feat: Dict[str, Any] = {}
    
    # Session metadata
    row_feat["session_id"] = str(window_df["session_id"].iloc[0])
    row_feat["window_start"] = window_df["timestamp"].min()
    row_feat["window_end"] = window_df["timestamp"].max()

    # Label extraction (majority label)
    labels = window_df["label"].dropna()
    if len(labels) > 0:
        row_feat["ground_truth_label"] = labels.mode().iloc[0]
    else:
        row_feat["ground_truth_label"] = None

    # GPS & Speed metadata (median/midpoint)
    valid_gps = window_df.dropna(subset=["latitude", "longitude"])
    if len(valid_gps) > 0:
        mid_idx = len(valid_gps) // 2
        row_feat["latitude"] = float(valid_gps["latitude"].iloc[mid_idx])
        row_feat["longitude"] = float(valid_gps["longitude"].iloc[mid_idx])
        row_feat["speed_kmh"] = float(valid_gps["speed_kmh"].median()) if "speed_kmh" in valid_gps else 0.0
        row_feat["gps_accuracy_m"] = float(valid_gps["gps_accuracy_m"].median()) if "gps_accuracy_m" in valid_gps else 0.0
    else:
        row_feat["latitude"] = np.nan
        row_feat["longitude"] = np.nan
        row_feat["speed_kmh"] = float(window_df["speed_kmh"].median()) if "speed_kmh" in window_df else 0.0
        row_feat["gps_accuracy_m"] = float(window_df["gps_accuracy_m"].median()) if "gps_accuracy_m" in window_df else 0.0

    # Vehicle type metadata if present
    if "vehicle_type" in window_df.columns and window_df["vehicle_type"].notna().any():
        row_feat["vehicle_type"] = str(window_df["vehicle_type"].dropna().iloc[0])
    else:
        row_feat["vehicle_type"] = "UNKNOWN"

    # Select signal prefix/channels depending on ablation stage
    use_calibrated = config_stage.upper() in ["C", "D", "E", "CALIBRATED", "CALIBRATED_GYRO", "CALIBRATED_GYRO_FFT"]
    use_filtered = config_stage.upper() in ["B", "FILTERED"]
    
    # Acc signal column mapping
    if use_calibrated and "acc_x_calibrated" in window_df.columns:
        ax_col, ay_col, az_col = "acc_x_calibrated", "acc_y_calibrated", "acc_z_calibrated"
        amag_col = "acc_magnitude_calibrated" if "acc_magnitude_calibrated" in window_df.columns else "acc_magnitude"
    elif use_filtered and "acc_x_raw_filtered" in window_df.columns:
        ax_col, ay_col, az_col = "acc_x", "acc_y", "acc_z"
        amag_col = "acc_magnitude"
    else:
        ax_col = "acc_x_raw_hampel" if "acc_x_raw_hampel" in window_df.columns else "acc_x"
        ay_col = "acc_y_raw_hampel" if "acc_y_raw_hampel" in window_df.columns else "acc_y"
        az_col = "acc_z_raw_hampel" if "acc_z_raw_hampel" in window_df.columns else "acc_z"
        amag_col = "acc_magnitude"

    # Time-domain acceleration features
    row_feat.update(compute_time_domain_features_1d(window_df[ax_col].values, "acc_x", include_full=True))
    row_feat.update(compute_time_domain_features_1d(window_df[ay_col].values, "acc_y", include_full=True))
    row_feat.update(compute_time_domain_features_1d(window_df[az_col].values, "acc_z", include_full=True))
    
    acc_mag_arr = window_df[amag_col].values if amag_col in window_df.columns else np.sqrt(window_df[ax_col]**2 + window_df[ay_col]**2 + window_df[az_col]**2).values
    row_feat.update(compute_time_domain_features_1d(acc_mag_arr, "acc_mag", include_full=True))

    # Jerk features
    jerk_col = "jerk_calibrated" if (use_calibrated and "jerk_calibrated" in window_df.columns) else "jerk"
    jerk_arr = window_df[jerk_col].values if jerk_col in window_df.columns else np.zeros(len(window_df))
    row_feat.update(compute_time_domain_features_1d(jerk_arr, "jerk", include_full=False))

    # Speed features
    speed_arr = window_df["speed_kmh"].dropna().values if "speed_kmh" in window_df.columns else np.zeros(len(window_df))
    row_feat.update(compute_time_domain_features_1d(speed_arr, "speed", include_full=False))

    # Impact Score stats
    impact_arr = window_df["impact_score"].values if "impact_score" in window_df.columns else np.zeros(len(window_df))
    row_feat["impact_score_mean"] = float(np.mean(impact_arr)) if len(impact_arr) > 0 else 0.0
    row_feat["impact_score_max"] = float(np.max(impact_arr)) if len(impact_arr) > 0 else 0.0
    row_feat["impact_score_std"] = float(np.std(impact_arr)) if len(impact_arr) > 0 else 0.0

    # Gyroscope features (included in D and E)
    if config_stage.upper() in ["D", "E", "CALIBRATED_GYRO", "CALIBRATED_GYRO_FFT", "CALIBRATED+GYRO"]:
        gx_col = "gyro_x_calibrated" if (use_calibrated and "gyro_x_calibrated" in window_df.columns) else "gyro_x"
        gy_col = "gyro_y_calibrated" if (use_calibrated and "gyro_y_calibrated" in window_df.columns) else "gyro_y"
        gz_col = "gyro_z_calibrated" if (use_calibrated and "gyro_z_calibrated" in window_df.columns) else "gyro_z"
        gmag_col = "gyro_magnitude_calibrated" if (use_calibrated and "gyro_magnitude_calibrated" in window_df.columns) else "gyro_magnitude"

        gx_arr = window_df[gx_col].values if gx_col in window_df.columns else np.zeros(len(window_df))
        gy_arr = window_df[gy_col].values if gy_col in window_df.columns else np.zeros(len(window_df))
        gz_arr = window_df[gz_col].values if gz_col in window_df.columns else np.zeros(len(window_df))
        gmag_arr = window_df[gmag_col].values if gmag_col in window_df.columns else np.zeros(len(window_df))

        row_feat.update(compute_time_domain_features_1d(gx_arr, "gyro_x", include_full=False))
        row_feat.update(compute_time_domain_features_1d(gy_arr, "gyro_y", include_full=False))
        row_feat.update(compute_time_domain_features_1d(gz_arr, "gyro_z", include_full=False))
        row_feat.update(compute_time_domain_features_1d(gmag_arr, "gyro_mag", include_full=False))

    # FFT features (included in E)
    if config_stage.upper() in ["E", "CALIBRATED_GYRO_FFT", "FFT"]:
        row_feat.update(compute_fft_features_1d(acc_mag_arr, sampling_rate_hz, prefix="acc_mag"))
        gmag_col = "gyro_magnitude_calibrated" if "gyro_magnitude_calibrated" in window_df.columns else "gyro_magnitude"
        gmag_arr = window_df[gmag_col].values if gmag_col in window_df.columns else np.zeros(len(window_df))
        row_feat.update(compute_fft_features_1d(gmag_arr, sampling_rate_hz, prefix="gyro_mag"))

    return row_feat

def extract_features(
    df_dsp: pd.DataFrame,
    config: Optional[AppConfig] = None,
    config_stage: str = "E"
) -> pd.DataFrame:
    """Extracts windowed features across all sessions in the DSP dataframe."""
    if config is None:
        config = load_config()

    duration_sec = config.window.duration_seconds
    target_hz = config.sampling.target_hz
    window_samples = int(np.round(duration_sec * target_hz))
    if window_samples < 5:
        window_samples = 5

    logger.info(f"Extracting features using {duration_sec}s windows ({window_samples} samples per window) for Stage {config_stage}...")

    window_rows: List[Dict[str, Any]] = []

    for sid, group in df_dsp.groupby("session_id"):
        group_sorted = group.sort_values("timestamp").reset_index(drop=True)
        n_samples = len(group_sorted)
        
        start_idx = 0
        while start_idx + window_samples <= n_samples:
            sub_window = group_sorted.iloc[start_idx : start_idx + window_samples]
            feat_dict = extract_features_from_window(sub_window, sampling_rate_hz=target_hz, config_stage=config_stage)
            window_rows.append(feat_dict)
            start_idx += window_samples

    if not window_rows:
        logger.warning("No complete windows could be extracted.")
        return pd.DataFrame()

    df_features = pd.DataFrame(window_rows)
    
    # Fill any missing values with 0.0 for numeric columns
    metadata_cols = ["session_id", "window_start", "window_end", "ground_truth_label", "latitude", "longitude", "vehicle_type"]
    feature_cols = [c for c in df_features.columns if c not in metadata_cols]
    
    df_features[feature_cols] = df_features[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    
    logger.info(f"Extracted {len(df_features)} window feature vectors with {len(feature_cols)} features each.")
    return df_features
