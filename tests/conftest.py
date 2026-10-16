import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def synthetic_csv_df():
    """Generates a synthetic smartphone dataset for unit testing."""
    n_samples = 600
    timestamps = pd.date_range("2026-09-26 10:00:00", periods=n_samples, freq="20ms", tz="UTC")
    elapsed = (timestamps - timestamps[0]).total_seconds().values

    # Generate synthetic sinusoidal signals with spikes
    t = elapsed
    acc_x = 0.5 * np.sin(2 * np.pi * 1.5 * t) + np.random.normal(0, 0.1, n_samples)
    acc_y = 0.3 * np.cos(2 * np.pi * 1.5 * t) + np.random.normal(0, 0.1, n_samples)
    acc_z = 9.81 + 1.2 * np.sin(2 * np.pi * 5.0 * t) + np.random.normal(0, 0.2, n_samples)
    
    # Add a spike for Hampel test
    acc_z[50] = 35.0

    gyro_x = 0.05 * np.sin(t)
    gyro_y = 0.04 * np.cos(t)
    gyro_z = 0.02 * np.sin(2 * t)

    lats = 18.5204 + 0.0001 * t
    lons = 73.8567 + 0.0001 * t
    speeds = 30.0 + np.random.normal(0, 1.0, n_samples)
    acc_m = np.full(n_samples, 10.0)

    # Synthetic labels
    labels = ["SMOOTH"] * 240 + ["POTHOLE"] * 120 + ["SPEED BREAKER"] * 120 + ["ROUGH"] * 120

    df = pd.DataFrame({
        "session_id": "TEST_SESS_01",
        "id": np.arange(1, n_samples + 1),
        "timestamp": timestamps,
        "elapsed_time_s": elapsed,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "latitude": lats,
        "longitude": lons,
        "speed_kmh": speeds,
        "gps_accuracy_m": acc_m,
        "label": labels
    })
    return df
