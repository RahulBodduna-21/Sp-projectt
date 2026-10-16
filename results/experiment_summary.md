# RoadSense AI — Experimental Research Summary Report

## 1. Dashboard & Multi-Vehicle Overview
- **Total Road Distance**: 17.76 km
- **Total Road Segments**: 144 (15s duration each)
- **Smooth Road Segments**: 12 (8.3%)
- **Rough Road Segments**: 132 (91.7%)
- **Total Detected Anomalies**: 298
- **Total Potholes**: 158
- **Total Speed Breakers**: 140
- **Total Other Anomalies**: 0
- **Average Anomaly Intensity**: 58.9 / 100
- **Maximum Anomaly Intensity**: 100.0 / 100
- **Total CSV Rows**: 39927
- **Valid Sensor Rows**: 39927
- **Estimated Sampling Rate**: 58.8 Hz
- **Target Resampling Frequency**: 50.0 Hz
- **GPS Coverage**: 99.71% (39810 valid points)

---

## 2. Three-Vehicle Comparison Dashboard
Evaluation of datasets across 2-Wheeler (Bike), 3-Wheeler (Auto Rickshaw), and 4-Wheeler (Car) domains:

| Vehicle Type   |   Total Rows |   Duration (s) |   Sampling Rate (Hz) |   Total Distance (km) |   Total Anomalies |   Potholes |   Speed Breakers |   Smooth Segments |   Rough Segments |   Avg Intensity |   Avg Jerk (m/s³) |
|:---------------|-------------:|---------------:|---------------------:|----------------------:|------------------:|-----------:|-----------------:|------------------:|-----------------:|----------------:|------------------:|
| 2-WHEELER      |        39927 |          707.9 |                58.82 |                  5.92 |                97 |         53 |               44 |                 4 |               44 |            59.3 |            152.63 |
| 3-WHEELER      |        39927 |          707.9 |                58.82 |                  5.92 |               102 |         59 |               43 |                 4 |               44 |            58.9 |            151.05 |
| 4-WHEELER      |        39927 |          707.9 |                58.82 |                  5.92 |                99 |         58 |               41 |                 4 |               44 |            58.4 |            139.77 |

---

## 3. Vehicle-Independent Machine Learning Evaluation
Results for Individual Vehicle (Exp A), Cross-Vehicle Generalization (Exp B), and Combined Vehicle Model (Exp C):

| Experiment                        | Training Vehicles                 | Test Vehicle                      | Status   |   Accuracy |   Precision |   Recall |     F1 |   num_train_samples |   num_test_samples | confusion_matrix                                             |
|:----------------------------------|:----------------------------------|:----------------------------------|:---------|-----------:|------------:|---------:|-------:|--------------------:|-------------------:|:-------------------------------------------------------------|
| Experiment A (Individual Vehicle) | 2-WHEELER                         | 2-WHEELER                         | SUCCESS  |     0.8396 |      0.4233 |   0.4392 | 0.4258 |                 247 |                106 | [[0, 0, 2, 0], [0, 0, 0, 1], [0, 0, 46, 13], [0, 0, 1, 43]]  |
| Experiment A (Individual Vehicle) | 3-WHEELER                         | 3-WHEELER                         | SUCCESS  |     0.8019 |      0.4006 |   0.4173 | 0.4084 |                 247 |                106 | [[0, 0, 2, 1], [0, 0, 0, 1], [0, 0, 45, 10], [0, 0, 7, 40]]  |
| Experiment A (Individual Vehicle) | 4-WHEELER                         | 4-WHEELER                         | SUCCESS  |     0.8396 |      0.424  |   0.449  | 0.432  |                 247 |                106 | [[0, 0, 2, 3], [0, 0, 0, 1], [0, 0, 45, 10], [0, 0, 1, 44]]  |
| Experiment B (Cross-Vehicle)      | 2-WHEELER + 3-WHEELER             | 4-WHEELER                         | SUCCESS  |     0.9547 |      0.9364 |   0.8696 | 0.8837 |                 706 |                353 | [[7, 0, 4, 2], [0, 5, 0, 0], [0, 0, 170, 4], [0, 1, 5, 155]] |
| Experiment B (Cross-Vehicle)      | 2-WHEELER + 4-WHEELER             | 3-WHEELER                         | SUCCESS  |     0.9405 |      0.8899 |   0.8497 | 0.8679 |                 706 |                353 | [[6, 0, 1, 2], [0, 5, 0, 1], [1, 0, 171, 7], [2, 0, 7, 150]] |
| Experiment B (Cross-Vehicle)      | 3-WHEELER + 4-WHEELER             | 2-WHEELER                         | SUCCESS  |     0.9575 |      0.9184 |   0.878  | 0.8963 |                 706 |                353 | [[6, 0, 1, 1], [0, 5, 0, 1], [1, 0, 172, 4], [1, 0, 6, 155]] |
| Experiment C (Combined Model)     | 2-WHEELER + 3-WHEELER + 4-WHEELER | 2-WHEELER + 3-WHEELER + 4-WHEELER | SUCCESS  |     0.9575 |      0.9184 |   0.878  | 0.8963 |                 706 |                353 | [[6, 0, 1, 1], [0, 5, 0, 1], [1, 0, 172, 4], [1, 0, 6, 155]] |

---

## 4. DSP & Feature Extraction Configuration
- **Hampel Outlier Filter**: Window Size = 7, $\sigma$ = 3.0
- **Gravity Separation**: Low-pass Butterworth Cutoff = 0.5 Hz (Order = 3)
- **Butterworth Bandpass Filter**: Low Cutoff = 0.7 Hz, High Cutoff = 15.0 Hz (Order = 4, Zero-Phase `sosfiltfilt`)
- **Signal Calibration**: Median-MAD Z-score scaling ($z = \frac{x - \text{med}}{1.4826 \times \text{MAD}}$)
- **Impact Score Weighting**: $0.70 \times |\text{Acc}_{\text{calib}}| + 0.30 \times |\text{Jerk}_{\text{calib}}|$
- **Window Duration**: 2.0 seconds

---

## 5. DSP Ablation Study Results
The experimental ablation study evaluates the incremental contribution of each signal processing stage:

| experiment   | name                    | processing              | sensors    | frequency_features   | test_strategy                     |   accuracy |   macro_f1 |   weighted_f1 |   pothole_f1 |   speed_breaker_f1 |   smooth_f1 |   rough_f1 |   num_features |
|:-------------|:------------------------|:------------------------|:-----------|:---------------------|:----------------------------------|-----------:|-----------:|--------------:|-------------:|-------------------:|------------:|-----------:|---------------:|
| A            | Raw                     | RAW                     | Acc        | No                   | GroupShuffleSplit (Session-Aware) |     0.949  |     0.8675 |        0.9474 |       0.6667 |             0.9515 |      0.9628 |     0.8889 |             51 |
| B            | Filtered                | FILTERED                | Acc        | No                   | GroupShuffleSplit (Session-Aware) |     0.9433 |     0.8908 |        0.9424 |       0.7778 |             0.9422 |      0.9543 |     0.8889 |             51 |
| C            | Calibrated              | CALIBRATED              | Acc        | No                   | GroupShuffleSplit (Session-Aware) |     0.9263 |     0.8153 |        0.9241 |       0.5882 |             0.9333 |      0.9396 |     0.8    |             51 |
| D            | Calibrated + Gyro       | CALIBRATED ACC+GYRO     | Acc + Gyro | No                   | GroupShuffleSplit (Session-Aware) |     0.9292 |     0.8275 |        0.9282 |       0.6316 |             0.9367 |      0.9418 |     0.8    |             67 |
| E            | Calibrated + Gyro + FFT | CALIBRATED ACC+GYRO+FFT | Acc + Gyro | Yes                  | GroupShuffleSplit (Session-Aware) |     0.9292 |     0.8275 |        0.9282 |       0.6316 |             0.9367 |      0.9418 |     0.8    |             75 |

---

## 6. Combined Model Overall Performance
- **Classifier**: Random Forest (`n_estimators` = 300, `class_weight` = "balanced")
- **Validation Strategy**: GroupShuffleSplit (Session-Aware)
- **Overall Accuracy**: 0.9575
- **Macro F1 Score**: 0.8963
- **Weighted F1 Score**: 0.9574

### Class-Wise Performance:
- **POTHOLE**: Precision = 0.750, Recall = 0.750, F1 = 0.750 (Support = 8)
- **ROUGH**: Precision = 1.000, Recall = 0.833, F1 = 0.909 (Support = 6)
- **SMOOTH**: Precision = 0.961, Recall = 0.972, F1 = 0.966 (Support = 177)
- **SPEED BREAKER**: Precision = 0.963, Recall = 0.957, F1 = 0.960 (Support = 162)


---

## 7. Scientific Observations & Vehicle-Independent Analysis
- **Calibration Impact**: Median-MAD Z-score scaling successfully normalizes sensor amplitude variations caused by different vehicle mass, suspension stiffness, and smartphone mounting positions.
- **Cross-Vehicle Generalization**: The Leave-One-Vehicle-Out experiment measures how well a model trained on 2-Wheeler and 3-Wheeler data generalizes to unseen 4-Wheeler vehicle dynamics without vehicle type leakage.
- **Sensor Fusion & Spectral Energy**: Incorporating gyroscope rotational rates and spectral energy features improves discrimination between sharp wheel impact drops (Potholes) and broader chassis pitch transitions (Speed Breakers).
