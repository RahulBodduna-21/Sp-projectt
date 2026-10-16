import numpy as np
from scipy.fft import rfft, rfftfreq
from typing import Dict

def compute_fft_features_1d(
    signal: np.ndarray,
    sampling_rate_hz: float = 50.0,
    prefix: str = "acc_mag"
) -> Dict[str, float]:
    """Computes frequency domain FFT features for a 1D signal."""
    features = {
        f"{prefix}_dominant_freq": 0.0,
        f"{prefix}_max_fft_magnitude": 0.0,
        f"{prefix}_spectral_energy": 0.0,
        f"{prefix}_spectral_centroid": 0.0
    }
    
    n = len(signal)
    if n < 4:
        return features

    clean_sig = np.nan_to_num(signal - np.mean(signal), nan=0.0, posinf=0.0, neginf=0.0)
    
    try:
        fft_vals = rfft(clean_sig)
        freqs = rfftfreq(n, d=1.0 / sampling_rate_hz)
        mags = np.abs(fft_vals)

        # Ignore DC component (freq > 0.1 Hz)
        ac_mask = freqs > 0.1
        if not np.any(ac_mask):
            return features

        ac_freqs = freqs[ac_mask]
        ac_mags = mags[ac_mask]

        max_idx = np.argmax(ac_mags)
        dom_freq = float(ac_freqs[max_idx])
        max_mag = float(ac_mags[max_idx])

        energy = float(np.sum(ac_mags**2) / len(ac_mags))
        
        sum_mags = np.sum(ac_mags)
        centroid = float(np.sum(ac_freqs * ac_mags) / sum_mags) if sum_mags > 0 else 0.0

        features[f"{prefix}_dominant_freq"] = dom_freq
        features[f"{prefix}_max_fft_magnitude"] = max_mag
        features[f"{prefix}_spectral_energy"] = energy
        features[f"{prefix}_spectral_centroid"] = centroid
    except Exception:
        pass

    return features
