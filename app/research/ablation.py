import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from app.core.config import AppConfig, load_config, get_project_root
from app.core.logging_config import logger
from app.dsp.pipeline import process_dsp_pipeline
from app.dsp.anomaly_detection import annotate_signal_labels
from app.features.extractor import extract_features
from app.ml.dataset_builder import build_ml_dataset
from app.ml.evaluator import evaluate_model

ABLATION_STAGES = [
    {"stage": "A", "name": "Raw", "sensors": "Acc", "freq": "No", "desc": "RAW -> FEATURES -> RF"},
    {"stage": "B", "name": "Filtered", "sensors": "Acc", "freq": "No", "desc": "FILTERED -> FEATURES -> RF"},
    {"stage": "C", "name": "Calibrated", "sensors": "Acc", "freq": "No", "desc": "CALIBRATED -> FEATURES -> RF"},
    {"stage": "D", "name": "Calibrated + Gyro", "sensors": "Acc + Gyro", "freq": "No", "desc": "CALIBRATED ACC+GYRO -> FEATURES -> RF"},
    {"stage": "E", "name": "Calibrated + Gyro + FFT", "sensors": "Acc + Gyro", "freq": "Yes", "desc": "CALIBRATED ACC+GYRO+FFT -> RF"}
]

def run_ablation_study(
    df_raw: pd.DataFrame,
    config: Optional[AppConfig] = None
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Runs automated 5-stage ablation study on input dataset."""
    if config is None:
        config = load_config()

    logger.info("================ STARTING ABLATION STUDY ================")
    
    # 1. Run DSP Pipeline once
    df_dsp, dsp_summary = process_dsp_pipeline(df_raw, config=config)
    df_dsp = annotate_signal_labels(df_dsp)
    
    results_list = []
    
    for item in ABLATION_STAGES:
        stage = item["stage"]
        name = item["name"]
        logger.info(f"--- Running Ablation Stage {stage}: {name} ---")
        
        try:
            df_feat = extract_features(df_dsp, config=config, config_stage=stage)
            if df_feat.empty:
                logger.warning(f"Stage {stage} produced empty features. Skipping.")
                continue

            X, y, groups, le, feat_names, df_valid = build_ml_dataset(df_feat)
            metrics, _, _ = evaluate_model(X, y, groups, le, config=config)

            pothole_f1 = metrics["per_class_metrics"].get("POTHOLE", {}).get("f1_score", np.nan)
            speed_breaker_f1 = metrics["per_class_metrics"].get("SPEED BREAKER", {}).get("f1_score", np.nan)
            smooth_f1 = metrics["per_class_metrics"].get("SMOOTH", {}).get("f1_score", np.nan)
            rough_f1 = metrics["per_class_metrics"].get("ROUGH", {}).get("f1_score", np.nan)

            row = {
                "experiment": stage,
                "name": name,
                "processing": item["desc"].split(" -> ")[0],
                "sensors": item["sensors"],
                "frequency_features": item["freq"],
                "test_strategy": metrics["test_strategy"],
                "accuracy": round(metrics["accuracy"], 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "weighted_f1": round(metrics["weighted_f1"], 4),
                "pothole_f1": round(pothole_f1, 4) if not np.isnan(pothole_f1) else "N/A",
                "speed_breaker_f1": round(speed_breaker_f1, 4) if not np.isnan(speed_breaker_f1) else "N/A",
                "smooth_f1": round(smooth_f1, 4) if not np.isnan(smooth_f1) else "N/A",
                "rough_f1": round(rough_f1, 4) if not np.isnan(rough_f1) else "N/A",
                "num_features": len(feat_names)
            }
            results_list.append(row)
            logger.info(f"Stage {stage} finished -> Accuracy: {row['accuracy']}, Macro F1: {row['macro_f1']}")
        except Exception as e:
            logger.error(f"Error in Ablation Stage {stage}: {e}")

    df_results = pd.DataFrame(results_list)
    
    # Save results table to results/ablation_results.csv
    results_dir = os.path.join(get_project_root(), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "ablation_results.csv")
    df_results.to_csv(out_csv, index=False)
    logger.info(f"Ablation study completed. Saved table to {out_csv}")
    
    return df_results, results_list
