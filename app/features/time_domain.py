import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from typing import Dict

def safe_stat(fn, arr, default=0.0):
    """Executes stat function safely returning default on empty/error."""
    if len(arr) == 0:
        return default
    try:
        val = fn(arr)
        return float(np.nan_to_num(val, nan=default, posinf=default, neginf=default))
    except Exception:
        return default

def compute_time_domain_features_1d(
    signal: np.ndarray,
    prefix: str,
    include_full: bool = True
) -> Dict[str, float]:
    """Computes time domain statistics for a 1D signal."""
    features = {}
    if len(signal) == 0:
        return features
        
    clean_sig = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    mean_val = safe_stat(np.mean, clean_sig)
    std_val = safe_stat(np.std, clean_sig)
    rms_val = safe_stat(lambda x: np.sqrt(np.mean(x**2)), clean_sig)
    max_val = safe_stat(np.max, clean_sig)
    
    features[f"{prefix}_mean"] = mean_val
    features[f"{prefix}_std"] = std_val
    features[f"{prefix}_rms"] = rms_val
    features[f"{prefix}_max"] = max_val

    if include_full:
        min_val = safe_stat(np.min, clean_sig)
        p2p_val = max_val - min_val
        med_val = safe_stat(np.median, clean_sig)
        mad_val = safe_stat(lambda x: np.median(np.abs(x - np.median(x))), clean_sig)
        skew_val = safe_stat(skew, clean_sig)
        kurt_val = safe_stat(kurtosis, clean_sig)

        features[f"{prefix}_min"] = min_val
        features[f"{prefix}_peak_to_peak"] = p2p_val
        features[f"{prefix}_median"] = med_val
        features[f"{prefix}_mad"] = mad_val
        features[f"{prefix}_skewness"] = skew_val
        features[f"{prefix}_kurtosis"] = kurt_val
        
    return features
