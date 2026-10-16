import pandas as pd
import numpy as np

def median_mad_calibration_1d(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Applies robust Median-MAD Z-score calibration to a 1D signal."""
    if len(x) == 0:
        return x.copy()
        
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    
    if mad <= eps or np.isnan(mad):
        # Fallback to standard deviation or zero-centering if MAD is zero
        std = np.std(x)
        if std > eps:
            z = (x - med) / std
        else:
            z = np.zeros_like(x)
    else:
        z = (x - med) / (1.4826 * mad)
        
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z

def apply_median_mad_calibration(
    df: pd.DataFrame,
    columns: list = None
) -> pd.DataFrame:
    """Applies Median-MAD calibration to specified channels in dataframe."""
    if columns is None:
        columns = ["acc_x", "acc_y", "acc_z", "acc_magnitude", "jerk", "gyro_x", "gyro_y", "gyro_z", "gyro_magnitude"]

    df_calib = df.copy()
    for col in columns:
        if col in df_calib.columns:
            calib_vals = np.zeros(len(df_calib))
            if "session_id" in df_calib.columns:
                for sid, group in df_calib.groupby("session_id"):
                    idx = group.index
                    calib_vals[idx] = median_mad_calibration_1d(group[col].values)
            else:
                calib_vals = median_mad_calibration_1d(df_calib[col].values)
            
            df_calib[f"{col}_calibrated"] = calib_vals
            
    return df_calib
