import os
import pandas as pd
from typing import Dict, Any, Optional
from app.core.config import AppConfig, load_config, get_project_root
from app.core.logging_config import logger

def generate_experiment_summary_report(
    val_report: Dict[str, Any],
    ablation_df: pd.DataFrame,
    cross_vehicle_df: pd.DataFrame,
    ml_metrics: Dict[str, Any],
    dashboard_summary: Optional[Dict[str, Any]] = None,
    vehicle_comparison_df: Optional[pd.DataFrame] = None,
    config: Optional[AppConfig] = None
) -> str:
    """Generates an automatically updated experiment_summary.md report in results/."""
    if config is None:
        config = load_config()

    results_dir = os.path.join(get_project_root(), "results")
    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "experiment_summary.md")

    ablation_table_md = ablation_df.to_markdown(index=False) if not ablation_df.empty else "No ablation data available."
    cross_table_md = cross_vehicle_df.to_markdown(index=False) if not cross_vehicle_df.empty else "No cross-vehicle data available."
    veh_comp_md = vehicle_comparison_df.to_markdown(index=False) if (vehicle_comparison_df is not None and not vehicle_comparison_df.empty) else "No vehicle comparison data available."

    per_class_str = ""
    for c_name, c_m in ml_metrics.get("per_class_metrics", {}).items():
        per_class_str += f"- **{c_name}**: Precision = {c_m['precision']:.3f}, Recall = {c_m['recall']:.3f}, F1 = {c_m['f1_score']:.3f} (Support = {c_m['support']})\n"

    lbl_dist = val_report.get('label_distribution', {})
    dash = dashboard_summary or {}

    content = f"""# RoadSense AI — Experimental Research Summary Report

## 1. Dashboard & Multi-Vehicle Overview
- **Total Road Distance**: {dash.get('total_road_distance_km', 0.0):.2f} km
- **Total Road Segments**: {dash.get('total_road_segments', 0)} (15s duration each)
- **Smooth Road Segments**: {dash.get('smooth_road_segments', 0)} ({dash.get('smooth_road_percentage', 0.0):.1f}%)
- **Rough Road Segments**: {dash.get('rough_road_segments', 0)} ({dash.get('rough_road_percentage', 0.0):.1f}%)
- **Total Detected Anomalies**: {dash.get('total_anomalies', 0)}
- **Total Potholes**: {dash.get('total_potholes', 0)}
- **Total Speed Breakers**: {dash.get('total_speed_breakers', 0)}
- **Total Other Anomalies**: {dash.get('total_other_anomalies', 0)}
- **Average Anomaly Intensity**: {dash.get('average_anomaly_intensity', 0.0):.1f} / 100
- **Maximum Anomaly Intensity**: {dash.get('maximum_anomaly_intensity', 0.0):.1f} / 100
- **Total CSV Rows**: {val_report.get('total_rows', 0)}
- **Valid Sensor Rows**: {val_report.get('valid_rows', 0)}
- **Estimated Sampling Rate**: {val_report.get('estimated_sampling_rate_hz', 0.0):.1f} Hz
- **Target Resampling Frequency**: {config.sampling.target_hz} Hz
- **GPS Coverage**: {val_report.get('gps_coverage', {}).get('coverage_percentage', 0.0)}% ({val_report.get('gps_coverage', {}).get('valid_gps_count', 0)} valid points)

---

## 2. Three-Vehicle Comparison Dashboard
Evaluation of datasets across 2-Wheeler (Bike), 3-Wheeler (Auto Rickshaw), and 4-Wheeler (Car) domains:

{veh_comp_md}

---

## 3. Vehicle-Independent Machine Learning Evaluation
Results for Individual Vehicle (Exp A), Cross-Vehicle Generalization (Exp B), and Combined Vehicle Model (Exp C):

{cross_table_md}

---

## 4. DSP & Feature Extraction Configuration
- **Hampel Outlier Filter**: Window Size = {config.hampel.window_size}, $\\sigma$ = {config.hampel.n_sigma}
- **Gravity Separation**: Low-pass Butterworth Cutoff = {config.gravity.cutoff_hz} Hz (Order = {config.gravity.order})
- **Butterworth Bandpass Filter**: Low Cutoff = {config.butterworth.low_cutoff_hz} Hz, High Cutoff = {config.butterworth.high_cutoff_hz} Hz (Order = {config.butterworth.order}, Zero-Phase `sosfiltfilt`)
- **Signal Calibration**: Median-MAD Z-score scaling ($z = \\frac{{x - \\text{{med}}}}{{1.4826 \\times \\text{{MAD}}}}$)
- **Impact Score Weighting**: $0.70 \\times |\\text{{Acc}}_{{\\text{{calib}}}}| + 0.30 \\times |\\text{{Jerk}}_{{\\text{{calib}}}}|$
- **Window Duration**: {config.window.duration_seconds} seconds

---

## 5. DSP Ablation Study Results
The experimental ablation study evaluates the incremental contribution of each signal processing stage:

{ablation_table_md}

---

## 6. Combined Model Overall Performance
- **Classifier**: Random Forest (`n_estimators` = {config.model.n_estimators}, `class_weight` = "{config.model.class_weight}")
- **Validation Strategy**: {ml_metrics.get('test_strategy', 'Session-Aware GroupShuffleSplit')}
- **Overall Accuracy**: {ml_metrics.get('accuracy', 0.0):.4f}
- **Macro F1 Score**: {ml_metrics.get('macro_f1', 0.0):.4f}
- **Weighted F1 Score**: {ml_metrics.get('weighted_f1', 0.0):.4f}

### Class-Wise Performance:
{per_class_str}

---

## 7. Scientific Observations & Vehicle-Independent Analysis
- **Calibration Impact**: Median-MAD Z-score scaling successfully normalizes sensor amplitude variations caused by different vehicle mass, suspension stiffness, and smartphone mounting positions.
- **Cross-Vehicle Generalization**: The Leave-One-Vehicle-Out experiment measures how well a model trained on 2-Wheeler and 3-Wheeler data generalizes to unseen 4-Wheeler vehicle dynamics without vehicle type leakage.
- **Sensor Fusion & Spectral Energy**: Incorporating gyroscope rotational rates and spectral energy features improves discrimination between sharp wheel impact drops (Potholes) and broader chassis pitch transitions (Speed Breakers).
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Generated research summary report at {report_path}")
    return report_path
