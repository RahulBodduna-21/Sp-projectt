import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt
from typing import Tuple, List

def separate_gravity_1d(
    signal: np.ndarray,
    sampling_rate_hz: float = 50.0,
    cutoff_hz: float = 0.5,
    order: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Separates slow gravity component from dynamic component using low-pass Butterworth filter."""
    if len(signal) < 15:
        # Fallback for very short signals
        gravity = np.full_like(signal, np.mean(signal))
        dynamic = signal - gravity
        return gravity, dynamic

    nyquist = sampling_rate_hz / 2.0
    normal_cutoff = min(cutoff_hz / nyquist, 0.99)
    if normal_cutoff <= 0:
        gravity = np.zeros_like(signal)
        return gravity, signal.copy()

    sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")
    try:
        gravity = sosfiltfilt(sos, signal)
    except Exception:
        gravity = np.full_like(signal, np.mean(signal))

    dynamic = signal - gravity
    return gravity, dynamic

def apply_gravity_separation(
    df: pd.DataFrame,
    acc_cols: List[str] = ["acc_x", "acc_y", "acc_z"],
    sampling_rate_hz: float = 50.0,
    cutoff_hz: float = 0.5,
    order: int = 3
) -> pd.DataFrame:
    """Applies gravity separation to accelerometer channels in dataframe."""
    df_result = df.copy()
    
    for col in acc_cols:
        if col in df_result.columns:
            grav_arr = np.zeros(len(df_result))
            dyn_arr = np.zeros(len(df_result))
            
            if "session_id" in df_result.columns:
                for sid, group in df_result.groupby("session_id"):
                    idx = group.index
                    g, d = separate_gravity_1d(group[col].values, sampling_rate_hz, cutoff_hz, order)
                    grav_arr[idx] = g
                    dyn_arr[idx] = d
            else:
                g, d = separate_gravity_1d(df_result[col].values, sampling_rate_hz, cutoff_hz, order)
                grav_arr = g
                dyn_arr = d

            df_result[f"{col}_gravity"] = grav_arr
            df_result[f"{col}_dynamic"] = dyn_arr
            # Use dynamic acceleration as the clean signal for subsequent filtering
            df_result[col] = dyn_arr

    return df_result
