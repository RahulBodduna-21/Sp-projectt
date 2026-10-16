import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

def calculate_jerk(df: pd.DataFrame, default_dt: float = 0.02) -> pd.DataFrame:
    """Calculates numerical jerk (rate of change of acceleration magnitude over time dt)."""
    df_jerk = df.copy()
    
    # Prefer calibrated acceleration magnitude if available, else raw magnitude
    if "acc_magnitude_calibrated" in df_jerk.columns:
        acc_signal_col = "acc_magnitude_calibrated"
    elif "acc_magnitude" in df_jerk.columns:
        acc_signal_col = "acc_magnitude"
    else:
        from app.dsp.magnitude import calculate_magnitudes
        df_jerk = calculate_magnitudes(df_jerk)
        acc_signal_col = "acc_magnitude"
        
    jerk_vals = np.zeros(len(df_jerk))
    
    if "session_id" in df_jerk.columns:
        for sid, group in df_jerk.groupby("session_id"):
            idx = group.index
            acc_sig = group[acc_signal_col].values
            
            if "timestamp" in group.columns and group["timestamp"].notna().sum() > 1:
                dt_sec = group["timestamp"].diff().dt.total_seconds().values
                dt_sec = np.nan_to_num(dt_sec, nan=default_dt)
                dt_sec[dt_sec <= 0] = default_dt
            elif "elapsed_time_s" in group.columns and group["elapsed_time_s"].notna().sum() > 1:
                dt_sec = np.diff(group["elapsed_time_s"].values, prepend=group["elapsed_time_s"].values[0])
                dt_sec = np.nan_to_num(dt_sec, nan=default_dt)
                dt_sec[dt_sec <= 0] = default_dt
            else:
                dt_sec = np.full(len(group), default_dt)

            d_acc = np.diff(acc_sig, prepend=acc_sig[0])
            jerk_group = d_acc / dt_sec
            jerk_group = np.nan_to_num(jerk_group, nan=0.0, posinf=0.0, neginf=0.0)
            jerk_vals[idx] = jerk_group
    else:
        acc_sig = df_jerk[acc_signal_col].values
        dt_sec = np.full(len(df_jerk), default_dt)
        d_acc = np.diff(acc_sig, prepend=acc_sig[0])
        jerk_vals = np.nan_to_num(d_acc / dt_sec, nan=0.0, posinf=0.0, neginf=0.0)

    df_jerk["jerk"] = jerk_vals
    return df_jerk

def compute_adaptive_jerk_threshold(
    jerk_signal: np.ndarray,
    k_multiplier: float = 3.0
) -> Tuple[float, float, float]:
    """Computes dataset-adaptive jerk anomaly threshold using Median-MAD statistics."""
    if len(jerk_signal) == 0:
        return 0.0, 0.0, 0.0
        
    abs_jerk = np.abs(np.nan_to_num(jerk_signal, nan=0.0, posinf=0.0, neginf=0.0))
    jerk_center = float(np.median(abs_jerk))
    jerk_mad = float(np.median(np.abs(abs_jerk - jerk_center)))

    if jerk_mad > 1e-6:
        threshold = jerk_center + k_multiplier * 1.4826 * jerk_mad
    else:
        # Fallback to 95th percentile or standard deviation if MAD is zero
        p95 = float(np.percentile(abs_jerk, 95))
        std_val = float(np.std(abs_jerk))
        threshold = p95 if p95 > 0 else (jerk_center + k_multiplier * std_val)

    return float(round(jerk_center, 4)), float(round(jerk_mad, 4)), float(round(threshold, 4))
