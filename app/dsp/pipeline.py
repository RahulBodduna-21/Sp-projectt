import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional

from app.core.config import AppConfig, load_config
from app.core.logging_config import logger
from app.dsp.validation import validate_dataset
from app.dsp.sampling import resample_dataset
from app.dsp.hampel import apply_hampel_filter
from app.dsp.gravity import apply_gravity_separation
from app.dsp.filtering import apply_butterworth_filter
from app.dsp.magnitude import calculate_magnitudes
from app.dsp.jerk import calculate_jerk
from app.dsp.calibration import apply_median_mad_calibration

def process_dsp_pipeline(
    df_raw: pd.DataFrame,
    config: Optional[AppConfig] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs the complete end-to-end DSP processing pipeline on input dataframe."""
    if config is None:
        config = load_config()

    logger.info("Starting DSP Pipeline execution...")
    
    # 1. Validation & Normalization
    df_valid, val_report = validate_dataset(df_raw)
    
    # 2. Resampling
    df_resampled, samp_stats = resample_dataset(
        df_valid,
        target_hz=config.sampling.target_hz
    )
    
    # 3. Hampel Outlier Removal
    df_hampel = apply_hampel_filter(
        df_resampled,
        window_size=config.hampel.window_size,
        n_sigma=config.hampel.n_sigma
    )
    
    # 4. Gravity / Slow Component Separation
    df_gravity = apply_gravity_separation(
        df_hampel,
        acc_cols=["acc_x", "acc_y", "acc_z"],
        sampling_rate_hz=config.sampling.target_hz,
        cutoff_hz=config.gravity.cutoff_hz,
        order=config.gravity.order
    )
    
    # 5. Butterworth Bandpass Filtering
    df_filtered = apply_butterworth_filter(
        df_gravity,
        columns=["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        sampling_rate_hz=config.sampling.target_hz,
        low_cutoff_hz=config.butterworth.low_cutoff_hz,
        high_cutoff_hz=config.butterworth.high_cutoff_hz,
        order=config.butterworth.order
    )
    
    # 6. Magnitudes calculation
    df_mag = calculate_magnitudes(df_filtered)
    
    # 7. Numerical Jerk calculation
    df_jerk = calculate_jerk(df_mag, default_dt=1.0 / config.sampling.target_hz)
    
    # 8. Median-MAD Calibration
    df_calib = apply_median_mad_calibration(
        df_jerk,
        columns=["acc_x", "acc_y", "acc_z", "acc_magnitude", "jerk", "gyro_x", "gyro_y", "gyro_z", "gyro_magnitude"]
    )
    
    # 9. Impact Score calculation
    acc_calib = df_calib["acc_magnitude_calibrated"].values if "acc_magnitude_calibrated" in df_calib.columns else np.zeros(len(df_calib))
    jerk_calib = df_calib["jerk_calibrated"].values if "jerk_calibrated" in df_calib.columns else np.zeros(len(df_calib))
    
    impact_score = 0.70 * np.abs(acc_calib) + 0.30 * np.abs(jerk_calib)
    df_calib["impact_score"] = np.nan_to_num(impact_score, nan=0.0, posinf=0.0, neginf=0.0)
    
    pipeline_summary = {
        "validation_report": val_report,
        "sampling_stats": samp_stats,
        "hampel_config": config.hampel.model_dump(),
        "gravity_config": config.gravity.model_dump(),
        "butterworth_config": config.butterworth.model_dump(),
        "target_hz": config.sampling.target_hz,
        "final_rows": len(df_calib)
    }
    
    logger.info(f"DSP Pipeline execution finished successfully. Output rows: {len(df_calib)}")
    return df_calib, pipeline_summary
