import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import numpy as np
import pandas as pd
from app.dsp.validation import normalize_label, validate_dataset
from app.mapping.gps import is_valid_gps
from app.dsp.sampling import estimate_sampling_rate, resample_dataset
from app.dsp.hampel import hampel_filter_1d, apply_hampel_filter
from app.dsp.filtering import butterworth_bandpass_1d, apply_butterworth_filter
from app.dsp.magnitude import calculate_magnitudes
from app.dsp.jerk import calculate_jerk
from app.dsp.calibration import median_mad_calibration_1d, apply_median_mad_calibration
from app.dsp.pipeline import process_dsp_pipeline
from app.features.extractor import extract_features
from app.features.frequency_domain import compute_fft_features_1d
from app.mapping.ground_truth import extract_ground_truth_anomalies
from app.ml.dataset_builder import build_ml_dataset
from app.ml.trainer import train_classifier
from app.ml.evaluator import evaluate_model
from app.ml.predictor import generate_predictions

def test_label_normalization():
    assert normalize_label("pothole") == "POTHOLE"
    assert normalize_label("Speed_Breaker") == "SPEED BREAKER"
    assert normalize_label(" speed-breaker ") == "SPEED BREAKER"
    assert normalize_label("smooth_road") == "SMOOTH"
    assert normalize_label(None) is None

def test_gps_validation():
    assert is_valid_gps(18.5204, 73.8567) is True
    assert is_valid_gps(-91.0, 73.8567) is False
    assert is_valid_gps(18.5204, 185.0) is False
    assert is_valid_gps(0.0, 0.0) is False

def test_sampling_rate_estimation(synthetic_csv_df):
    sr, reg = estimate_sampling_rate(synthetic_csv_df["timestamp"])
    assert 45.0 <= sr <= 55.0
    assert reg > 0.8

def test_hampel_filter():
    sig = np.array([1.0, 1.1, 0.9, 1.0, 100.0, 1.0, 0.9, 1.1])
    filtered = hampel_filter_1d(sig, window_size=5, n_sigma=3.0)
    assert filtered[4] < 10.0

def test_butterworth_filter():
    sig = np.random.normal(0, 1, 100)
    filt = butterworth_bandpass_1d(sig, sampling_rate_hz=50.0, low_cutoff_hz=0.7, high_cutoff_hz=15.0)
    assert len(filt) == len(sig)
    assert not np.isnan(filt).any()

def test_magnitude_and_jerk(synthetic_csv_df):
    df_mag = calculate_magnitudes(synthetic_csv_df)
    assert "acc_magnitude" in df_mag.columns
    assert "gyro_magnitude" in df_mag.columns

    df_jerk = calculate_jerk(df_mag)
    assert "jerk" in df_jerk.columns
    assert not np.isnan(df_jerk["jerk"]).any()

def test_median_mad_calibration():
    sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    z = median_mad_calibration_1d(sig)
    assert not np.isnan(z).any()
    assert not np.isinf(z).any()

def test_fft_features():
    sig = np.sin(2 * np.pi * 5.0 * np.linspace(0, 1, 50))
    fft_feat = compute_fft_features_1d(sig, sampling_rate_hz=50.0)
    assert "acc_mag_dominant_freq" in fft_feat
    assert 4.0 <= fft_feat["acc_mag_dominant_freq"] <= 6.0

def test_ground_truth_extraction(synthetic_csv_df):
    df_gt, stats = extract_ground_truth_anomalies(synthetic_csv_df)
    assert stats["total_csv_rows"] == len(synthetic_csv_df)
    assert len(df_gt) > 0

def test_full_pipeline_and_ml(synthetic_csv_df):
    df_dsp, summary = process_dsp_pipeline(synthetic_csv_df)
    assert len(df_dsp) > 0

    df_feat = extract_features(df_dsp, config_stage="E")
    assert len(df_feat) > 0
    assert not df_feat.isna().any().any()

    X, y, groups, le, feat_names, df_valid = build_ml_dataset(df_feat)
    metrics, y_test, y_pred = evaluate_model(X, y, groups, le)
    assert metrics["accuracy"] >= 0.0

    model = train_classifier(X, y)
    df_preds = generate_predictions(model, le, df_feat, feat_names)
    assert len(df_preds) == len(df_feat)
    assert "confidence" in df_preds.columns

def test_three_vehicle_validation_and_cross_vehicle(synthetic_csv_df):
    from app.research.cross_vehicle import run_cross_vehicle_study
    df_2w, rep_2w = validate_dataset(synthetic_csv_df, default_vehicle_type="2-WHEELER")
    assert rep_2w["vehicle_type"] == "2-WHEELER"
    assert rep_2w["total_rows"] == len(synthetic_csv_df)

    df_3w = synthetic_csv_df.copy()
    df_3w["session_id"] = "SESS_3W_TEST"
    df_3w, rep_3w = validate_dataset(df_3w, default_vehicle_type="3-WHEELER")
    assert rep_3w["vehicle_type"] == "3-WHEELER"

    df_4w = synthetic_csv_df.copy()
    df_4w["session_id"] = "SESS_4W_TEST"
    df_4w, rep_4w = validate_dataset(df_4w, default_vehicle_type="4-WHEELER")
    assert rep_4w["vehicle_type"] == "4-WHEELER"

    veh_dfs = {"2-WHEELER": df_2w, "3-WHEELER": df_3w, "4-WHEELER": df_4w}
    df_res, info = run_cross_vehicle_study(veh_dfs)
    assert not df_res.empty
    assert "Accuracy" in df_res.columns
    assert "F1" in df_res.columns
