# RoadSense AI — Real-World Python Research Prototype

**RoadSense AI — Vehicle-Independent Road Anomaly Classification Through DSP-Based Signal Calibration Across Diverse Vehicle and Road Conditions**

---

## 1. Research Overview & Research Question

This research system investigates:

> *Can DSP-based signal calibration reduce vehicle-dependent variation in smartphone accelerometer and gyroscope signals and improve road-anomaly classification across different vehicles and road conditions?*

The system performs actual digital signal processing (DSP), windowed time/frequency domain feature extraction, session-aware machine learning (Random Forest), automated 5-stage ablation studies, leave-one-vehicle-out generalization experiments, independent ground-truth anomaly parsing, and interactive GPS map visualizations on real smartphone telemetry datasets.

---

## 2. System Architecture

```text
roadsense-ai/
│
├── app/
│   ├── main.py                     # FastAPI application entrypoint
│   ├── api/                        # REST API endpoints (dataset, dsp, features, ml, predictions, map, research)
│   ├── core/                       # App configuration, logging & exception handling
│   ├── dsp/                        # DSP Pipeline (validation, sampling, Hampel, gravity, Butterworth, magnitude, jerk, calibration)
│   ├── features/                   # Feature Extraction (time domain, FFT frequency domain, windowing)
│   ├── ml/                         # Machine Learning (dataset builder, trainer, evaluator, predictor, registry)
│   ├── mapping/                    # Mapping Engine (ground truth extraction, GPS validation, Folium map builder)
│   └── research/                   # Research Engine (ablation study, cross-vehicle experiment, plot metrics, report)
│
├── config.yaml                     # Central YAML configuration
├── data/                           # Data workspace (raw, processed, features, predictions)
├── models/                         # Trained joblib models & metadata
├── results/                        # Exported research tables, figures, metrics & interactive HTML map
├── scripts/                        # Executable CLI scripts (run_pipeline, run_ablation, run_cross_vehicle, train, evaluate)
├── notebooks/                      # Jupyter research notebooks (01_... to 07_...)
├── tests/                          # Automated Pytest unit test suite
├── requirements.txt
└── README.md
```

---

## 3. CSV Input Specification

Required columns:
```text
session_id, id, timestamp, elapsed_time_s, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, latitude, longitude, speed_kmh, gps_accuracy_m, label
```

Units:
- `acc_x/y/z`: $\text{m/s}^2$
- `gyro_x/y/z`: $\text{rad/s}$
- `latitude/longitude`: WGS84 degrees
- `speed_kmh`: $\text{km/h}$
- `gps_accuracy_m`: meters
- `timestamp`: UTC ISO timestamp

Recognized normalized classes: `SMOOTH`, `ROUGH`, `POTHOLE`, `SPEED BREAKER`.

---

## 4. Execution Commands

### End-to-End Pipeline Execution
```bash
python scripts/run_pipeline.py --input data/raw/dataset.csv
```

### 5-Stage Ablation Study Experiment
```bash
python scripts/run_ablation.py --input data/raw/dataset.csv
```

### Leave-One-Vehicle-Out Cross-Vehicle Generalization Experiment
```bash
python scripts/run_cross_vehicle.py --input data/raw/
```

### Model Training & Evaluation
```bash
python scripts/train.py --input data/raw/dataset.csv
python scripts/evaluate.py --input data/raw/dataset.csv
```

### Automated Pytest Suite
```bash
pytest tests/
```

### REST API Web Server
```bash
uvicorn app.main:app --reload --port 8000
```
Interactive API documentation: `http://127.0.0.1:8000/docs`

---

## 5. Experimental Ablation Configurations

| Stage | Processing | Sensors | Frequency Features (FFT) | Test Strategy |
|---|---|---|---|---|
| **A** | Raw | Accelerometer | No | Session-Aware / Stratified Split |
| **B** | Filtered (Butterworth 0.7–15 Hz) | Accelerometer | No | Session-Aware / Stratified Split |
| **C** | Calibrated (Median-MAD Z-score) | Accelerometer | No | Session-Aware / Stratified Split |
| **D** | Calibrated | Acc + Gyroscope | No | Session-Aware / Stratified Split |
| **E** | Calibrated | Acc + Gyroscope | Yes (scipy.fft) | Session-Aware / Stratified Split |
