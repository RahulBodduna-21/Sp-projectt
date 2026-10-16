from typing import List, Dict, Any
import pandas as pd

def get_ablation_feature_columns(config_stage: str) -> List[str]:
    """Returns the list of feature column names corresponding to an ablation stage."""
    # Config A: RAW acc
    acc_time_cols = [
        "acc_x_mean", "acc_x_std", "acc_x_rms", "acc_x_max", "acc_x_min", "acc_x_peak_to_peak", "acc_x_median", "acc_x_mad", "acc_x_skewness", "acc_x_kurtosis",
        "acc_y_mean", "acc_y_std", "acc_y_rms", "acc_y_max", "acc_y_min", "acc_y_peak_to_peak", "acc_y_median", "acc_y_mad", "acc_y_skewness", "acc_y_kurtosis",
        "acc_z_mean", "acc_z_std", "acc_z_rms", "acc_z_max", "acc_z_min", "acc_z_peak_to_peak", "acc_z_median", "acc_z_mad", "acc_z_skewness", "acc_z_kurtosis",
        "acc_mag_mean", "acc_mag_std", "acc_mag_rms", "acc_mag_max", "acc_mag_min", "acc_mag_peak_to_peak", "acc_mag_median", "acc_mag_mad", "acc_mag_skewness", "acc_mag_kurtosis",
        "jerk_mean", "jerk_std", "jerk_rms", "jerk_max",
        "speed_mean", "speed_std", "speed_min", "speed_max"
    ]
    
    if config_stage.upper() in ["A", "RAW"]:
        return acc_time_cols

    if config_stage.upper() in ["B", "FILTERED"]:
        return acc_time_cols

    if config_stage.upper() in ["C", "CALIBRATED"]:
        return acc_time_cols + ["impact_score_mean", "impact_score_max", "impact_score_std"]

    gyro_time_cols = [
        "gyro_x_mean", "gyro_x_std", "gyro_x_rms", "gyro_x_min", "gyro_x_max", "gyro_x_peak_to_peak",
        "gyro_y_mean", "gyro_y_std", "gyro_y_rms", "gyro_y_min", "gyro_y_max", "gyro_y_peak_to_peak",
        "gyro_z_mean", "gyro_z_std", "gyro_z_rms", "gyro_z_min", "gyro_z_max", "gyro_z_peak_to_peak",
        "gyro_mag_mean", "gyro_mag_std", "gyro_mag_rms", "gyro_mag_min", "gyro_mag_max", "gyro_mag_peak_to_peak"
    ]

    if config_stage.upper() in ["D", "CALIBRATED_GYRO", "CALIBRATED+GYRO"]:
        return acc_time_cols + ["impact_score_mean", "impact_score_max", "impact_score_std"] + gyro_time_cols

    fft_cols = [
        "acc_mag_dominant_freq", "acc_mag_max_fft_magnitude", "acc_mag_spectral_energy", "acc_mag_spectral_centroid",
        "gyro_mag_dominant_freq", "gyro_mag_max_fft_magnitude", "gyro_mag_spectral_energy", "gyro_mag_spectral_centroid"
    ]

    # Config E: Full pipeline (Calibrated + Gyro + FFT)
    return acc_time_cols + ["impact_score_mean", "impact_score_max", "impact_score_std"] + gyro_time_cols + fft_cols
