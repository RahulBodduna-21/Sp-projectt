import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from app.core.logging_config import logger

def estimate_sampling_rate(timestamps: pd.Series) -> Tuple[float, float]:
    """Estimates sampling rate (Hz) and regularity from timestamps series."""
    if timestamps is None or len(timestamps) < 2:
        return 50.0, 1.0
    
    dt = timestamps.diff().dt.total_seconds()
    dt_valid = dt[dt > 0].dropna()
    if len(dt_valid) == 0:
        return 50.0, 1.0
    
    median_dt = float(dt_valid.median())
    if median_dt <= 0:
        return 50.0, 1.0
    
    sr = 1.0 / median_dt
    std_dt = float(dt_valid.std()) if len(dt_valid) > 1 else 0.0
    regularity = float(1.0 - min(1.0, std_dt / median_dt)) if median_dt > 0 else 0.0
    return sr, regularity

def resample_session_signals(
    session_df: pd.DataFrame,
    target_hz: float = 50.0
) -> pd.DataFrame:
    """Resamples continuous sensor signals for a single session to target_hz using time interpolation."""
    if session_df.empty:
        return session_df

    df = session_df.copy().sort_values("timestamp").reset_index(drop=True)
    
    # Handle potential duplicate timestamps
    if df["timestamp"].duplicated().any():
        df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)

    start_time = df["timestamp"].min()
    end_time = df["timestamp"].max()
    if pd.isna(start_time) or pd.isna(end_time) or start_time >= end_time:
        return df

    # Create target regular timestamp grid
    total_seconds = (end_time - start_time).total_seconds()
    num_samples = int(np.round(total_seconds * target_hz)) + 1
    if num_samples <= 1:
        return df

    resampled_timestamps = pd.date_range(start=start_time, end=end_time, periods=num_samples, tz="UTC")
    resampled_elapsed = (resampled_timestamps - start_time).total_seconds()

    sensor_cols = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

    # Set timestamp as index for time interpolation
    df_indexed = df.set_index("timestamp")
    
    # Reindex to union of old and new timestamps to interpolate accurately
    combined_index = df_indexed.index.union(resampled_timestamps).sort_values().drop_duplicates()
    df_reindexed = df_indexed.reindex(combined_index)

    # Interpolate sensor continuous signals
    for col in sensor_cols:
        if col in df_reindexed.columns:
            df_reindexed[col] = df_reindexed[col].interpolate(method="time")

    # Extract only the target resampled grid
    df_resampled = df_reindexed.loc[resampled_timestamps].copy().reset_index()
    df_resampled.rename(columns={"index": "timestamp"}, inplace=True)
    df_resampled["elapsed_time_s"] = resampled_elapsed

    # Nearest neighbor mapping for non-interpolated fields (GPS, label, session_id, id)
    non_sensor_cols = ["session_id", "id", "label", "latitude", "longitude", "speed_kmh", "gps_accuracy_m"]
    orig_times = df["timestamp"].astype("int64").values
    target_times = df_resampled["timestamp"].astype("int64").values
    nearest_indices = np.searchsorted(orig_times, target_times, side="left")
    nearest_indices = np.clip(nearest_indices, 0, len(df) - 1)
    
    # Adjust for closest boundary
    for i, idx in enumerate(nearest_indices):
        if idx > 0:
            diff_curr = abs(target_times[i] - orig_times[idx])
            diff_prev = abs(target_times[i] - orig_times[idx - 1])
            if diff_prev < diff_curr:
                nearest_indices[i] = idx - 1

    for col in non_sensor_cols:
        if col in df.columns:
            df_resampled[col] = df[col].iloc[nearest_indices].values

    return df_resampled

def resample_dataset(df: pd.DataFrame, target_hz: float = 50.0) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Resamples multi-session dataset to target_hz."""
    orig_sr, regularity = estimate_sampling_rate(df["timestamp"])
    
    resampled_sessions = []
    for sid, group in df.groupby("session_id"):
        res_group = resample_session_signals(group, target_hz=target_hz)
        resampled_sessions.append(res_group)
    
    if resampled_sessions:
        df_resampled = pd.concat(resampled_sessions, ignore_index=True)
    else:
        df_resampled = df.copy()

    stats = {
        "original_sampling_rate_hz": round(orig_sr, 2),
        "target_sampling_rate_hz": target_hz,
        "sampling_regularity": round(regularity, 4),
        "original_rows": len(df),
        "resampled_rows": len(df_resampled)
    }
    logger.info(f"Resampling completed: {stats['original_rows']} -> {stats['resampled_rows']} rows @ {target_hz} Hz")
    return df_resampled, stats
