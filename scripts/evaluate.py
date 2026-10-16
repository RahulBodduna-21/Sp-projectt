import argparse
import os
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import load_config
from app.core.logging_config import logger
from app.dsp.pipeline import process_dsp_pipeline
from app.features.extractor import extract_features
from app.ml.dataset_builder import build_ml_dataset
from app.ml.evaluator import evaluate_model

def main():
    parser = argparse.ArgumentParser(description="RoadSense AI — Evaluate Classifier Model")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input smartphone CSV dataset")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to custom config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    df_raw = pd.read_csv(args.input)
    df_dsp, dsp_summary = process_dsp_pipeline(df_raw, config=config)
    df_features = extract_features(df_dsp, config=config, config_stage="E")

    X, y, groups, le, feat_names, df_valid = build_ml_dataset(df_features)
    metrics, y_test, y_pred = evaluate_model(X, y, groups, le, config=config)

    print("\n================ EVALUATION METRICS ================")
    print(f"Test Strategy: {metrics['test_strategy']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score: {metrics['weighted_f1']:.4f}")
    print("\nClassification Report:")
    print(metrics["classification_report_text"])
    print("===================================================\n")

if __name__ == "__main__":
    main()
