import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt, sosfilt
from typing import List, Optional
from app.core.logging_config import logger

def butterworth_bandpass_1d(
    signal: np.ndarray,
    sampling_rate_hz: float = 50.0,
    low_cutoff_hz: float = 0.7,
    high_cutoff_hz: float = 15.0,
    order: int = 4
) -> np.ndarray:
    """Applies Butterworth bandpass filter safely using zero-phase sosfiltfilt."""
    if len(signal) < 15:
        return signal.copy()

    nyquist = sampling_rate_hz / 2.0
    if nyquist <= 0:
        return signal.copy()

    low = max(0.01, low_cutoff_hz)
    high = min(high_cutoff_hz, nyquist * 0.95)

    if low >= high:
        low = min(0.1, nyquist * 0.1)
        high = nyquist * 0.90

    low_norm = low / nyquist
    high_norm = high / nyquist

    try:
        sos = butter(order, [low_norm, high_norm], btype="bandpass", output="sos")
        # Pad len check for sosfiltfilt
        padlen = 3 * (2 * order)
        if len(signal) > padlen:
            filtered = sosfiltfilt(sos, signal)
        else:
            filtered = sosfilt(sos, signal)
        return filtered
    except Exception as e:
        logger.warning(f"Butterworth filtering warning: {e}. Returning unfiltered signal.")
        return signal.copy()

def apply_butterworth_filter(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    sampling_rate_hz: float = 50.0,
    low_cutoff_hz: float = 0.7,
    high_cutoff_hz: float = 15.0,
    order: int = 4
) -> pd.DataFrame:
    """Applies Butterworth bandpass filter to specified sensor channels in dataframe."""
    if columns is None:
        columns = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

    df_filtered = df.copy()
    for col in columns:
        if col in df_filtered.columns:
            df_filtered[f"{col}_raw_filtered"] = df_filtered[col].copy()
            filtered_vals = np.zeros(len(df_filtered))
            
            if "session_id" in df_filtered.columns:
                for sid, group in df_filtered.groupby("session_id"):
                    idx = group.index
                    filtered_vals[idx] = butterworth_bandpass_1d(
                        group[col].values,
                        sampling_rate_hz=sampling_rate_hz,
                        low_cutoff_hz=low_cutoff_hz,
                        high_cutoff_hz=high_cutoff_hz,
                        order=order
                    )
                df_filtered[col] = filtered_vals
            else:
                df_filtered[col] = butterworth_bandpass_1d(
                    df_filtered[col].values,
                    sampling_rate_hz=sampling_rate_hz,
                    low_cutoff_hz=low_cutoff_hz,
                    high_cutoff_hz=high_cutoff_hz,
                    order=order
                )
    return df_filtered
