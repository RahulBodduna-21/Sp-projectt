import pandas as pd
import numpy as np
from typing import List, Optional

def hampel_filter_1d(
    signal: np.ndarray,
    window_size: int = 7,
    n_sigma: float = 3.0
) -> np.ndarray:
    """Applies Hampel filter to a 1D numpy array to remove outlier spikes."""
    if len(signal) < window_size:
        return signal.copy()
    
    n = len(signal)
    filtered = signal.copy()
    k = window_size // 2
    
    # Vectorized rolling window using pandas rolling for performance
    s_series = pd.Series(signal)
    rolling_median = s_series.rolling(window=window_size, center=True, min_periods=1).median().values
    
    # Compute rolling MAD
    abs_diff = np.abs(signal - rolling_median)
    rolling_mad = pd.Series(abs_diff).rolling(window=window_size, center=True, min_periods=1).median().values
    
    threshold = n_sigma * 1.4826 * rolling_mad
    outliers = np.abs(signal - rolling_median) > threshold
    
    # Replace outliers with rolling median
    filtered[outliers] = rolling_median[outliers]
    return filtered

def apply_hampel_filter(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    window_size: int = 7,
    n_sigma: float = 3.0
) -> pd.DataFrame:
    """Applies Hampel filter to specified sensor columns in dataframe."""
    if columns is None:
        columns = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    
    df_cleaned = df.copy()
    for col in columns:
        if col in df_cleaned.columns:
            # Save original signal for visualization comparison
            df_cleaned[f"{col}_raw_hampel"] = df_cleaned[col].copy()
            # Apply per session if session_id exists
            if "session_id" in df_cleaned.columns:
                cleaned_vals = np.zeros(len(df_cleaned))
                for sid, group in df_cleaned.groupby("session_id"):
                    idx = group.index
                    cleaned_vals[idx] = hampel_filter_1d(group[col].values, window_size=window_size, n_sigma=n_sigma)
                df_cleaned[col] = cleaned_vals
            else:
                df_cleaned[col] = hampel_filter_1d(df_cleaned[col].values, window_size=window_size, n_sigma=n_sigma)
    return df_cleaned
