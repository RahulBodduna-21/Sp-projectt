import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from app.core.config import AppConfig, load_config, get_project_root
from app.core.logging_config import logger
from app.dsp.pipeline import process_dsp_pipeline
from app.dsp.anomaly_detection import annotate_signal_labels
from app.features.extractor import extract_features
from app.ml.dataset_builder import build_ml_dataset
from app.ml.trainer import train_classifier

CROSS_VEHICLE_CONFIGS = [
    {"exp_type": "Experiment A (Individual Vehicle)", "train": ["2-WHEELER"], "test": "2-WHEELER"},
    {"exp_type": "Experiment A (Individual Vehicle)", "train": ["3-WHEELER"], "test": "3-WHEELER"},
    {"exp_type": "Experiment A (Individual Vehicle)", "train": ["4-WHEELER"], "test": "4-WHEELER"},
    {"exp_type": "Experiment B (Cross-Vehicle)", "train": ["2-WHEELER", "3-WHEELER"], "test": "4-WHEELER"},
    {"exp_type": "Experiment B (Cross-Vehicle)", "train": ["2-WHEELER", "4-WHEELER"], "test": "3-WHEELER"},
    {"exp_type": "Experiment B (Cross-Vehicle)", "train": ["3-WHEELER", "4-WHEELER"], "test": "2-WHEELER"},
    {"exp_type": "Experiment C (Combined Model)", "train": ["2-WHEELER", "3-WHEELER", "4-WHEELER"], "test": "2-WHEELER + 3-WHEELER + 4-WHEELER"}
]

def run_cross_vehicle_study(
    vehicle_datasets: Dict[str, pd.DataFrame],
    config: Optional[AppConfig] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs Experiments A, B, and C for vehicle-independent robustness evaluation."""
    if config is None:
        config = load_config()

    logger.info("================ STARTING THREE-VEHICLE CROSS-VEHICLE EXPERIMENTS ================")
    
    # Process DSP & Feature extraction per vehicle dataset
    vehicle_features = {}
    vehicle_dsp = {}
    
    for v_type, df_raw in vehicle_datasets.items():
        if df_raw is not None and not df_raw.empty:
            logger.info(f"Processing DSP & Feature Extraction for vehicle domain: {v_type}...")
            df_dsp, _ = process_dsp_pipeline(df_raw, config=config)
            df_dsp = annotate_signal_labels(df_dsp)
            df_dsp["vehicle_type"] = v_type
            df_feat = extract_features(df_dsp, config=config, config_stage="E")
            if not df_feat.empty:
                df_feat["vehicle_type"] = v_type
                vehicle_features[v_type] = df_feat
                vehicle_dsp[v_type] = df_dsp

    results_list = []

    for item in CROSS_VEHICLE_CONFIGS:
        exp_type = item["exp_type"]
        train_types = item["train"]
        test_type = item["test"]
        
        train_dfs = [vehicle_features[v] for v in train_types if v in vehicle_features]
        
        if test_type == "2-WHEELER + 3-WHEELER + 4-WHEELER":
            test_dfs = [vehicle_features[v] for v in vehicle_features]
            test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else None
        else:
            test_df = vehicle_features.get(test_type)

        if not train_dfs or test_df is None or test_df.empty:
            results_list.append({
                "Experiment": exp_type,
                "Training Vehicles": " + ".join(train_types),
                "Test Vehicle": test_type,
                "Status": "INSUFFICIENT_DATA",
                "Accuracy": "N/A",
                "Precision": "N/A",
                "Recall": "N/A",
                "F1": "N/A"
            })
            continue

        try:
            if exp_type == "Experiment A (Individual Vehicle)" or test_type == "2-WHEELER + 3-WHEELER + 4-WHEELER":
                # Session-aware train/test split to prevent leakage
                df_comb = pd.concat(train_dfs, ignore_index=True)
                X_all, y_all, groups, le, feat_cols, _ = build_ml_dataset(df_comb)
                
                if len(np.unique(groups)) > 1:
                    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
                    train_idx, test_idx = next(gss.split(X_all, y_all, groups))
                    X_tr, y_tr = X_all[train_idx], y_all[train_idx]
                    X_te, y_te = X_all[test_idx], y_all[test_idx]
                else:
                    split_idx = int(len(X_all) * 0.70)
                    X_tr, y_tr = X_all[:split_idx], y_all[:split_idx]
                    X_te, y_te = X_all[split_idx:], y_all[split_idx:]
            else:
                # Experiment B: Cross-vehicle (Train on train_types, Test on test_type)
                df_train_comb = pd.concat(train_dfs, ignore_index=True)
                X_tr, y_tr, _, le, feat_cols, _ = build_ml_dataset(df_train_comb)
                X_te_full, y_te_full, _, _, _, _ = build_ml_dataset(test_df, feature_columns=feat_cols, label_encoder=le)
                
                valid_te_mask = y_te_full >= 0
                X_te, y_te = X_te_full[valid_te_mask], y_te_full[valid_te_mask]

            if len(X_tr) == 0 or len(X_te) == 0:
                results_list.append({
                    "Experiment": exp_type,
                    "Training Vehicles": " + ".join(train_types),
                    "Test Vehicle": test_type,
                    "Status": "EMPTY_SPLIT",
                    "Accuracy": "N/A",
                    "Precision": "N/A",
                    "Recall": "N/A",
                    "F1": "N/A"
                })
                continue

            clf = train_classifier(X_tr, y_tr, config=config)
            y_pred = clf.predict(X_te)

            acc = float(accuracy_score(y_te, y_pred))
            prec, rec, f1_macro, _ = precision_recall_fscore_support(y_te, y_pred, average="macro", zero_division=0)
            cm = confusion_matrix(y_te, y_pred)

            row = {
                "Experiment": exp_type,
                "Training Vehicles": " + ".join(train_types),
                "Test Vehicle": test_type,
                "Status": "SUCCESS",
                "Accuracy": round(acc, 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "F1": round(float(f1_macro), 4),
                "num_train_samples": len(X_tr),
                "num_test_samples": len(X_te),
                "confusion_matrix": cm.tolist()
            }
            results_list.append(row)
            logger.info(f"{exp_type} ({' + '.join(train_types)} -> {test_type}) | Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1_macro:.4f}")
        except Exception as e:
            logger.error(f"Error in cross-vehicle experiment ({exp_type}): {e}")
            results_list.append({
                "Experiment": exp_type,
                "Training Vehicles": " + ".join(train_types),
                "Test Vehicle": test_type,
                "Status": f"ERROR: {str(e)}",
                "Accuracy": "N/A",
                "Precision": "N/A",
                "Recall": "N/A",
                "F1": "N/A"
            })

    df_results = pd.DataFrame(results_list)
    results_dir = os.path.join(get_project_root(), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "cross_vehicle_results.csv")
    df_results.to_csv(out_csv, index=False)
    logger.info(f"Three-vehicle cross-vehicle study completed. Saved results table to {out_csv}")

    return df_results, {"status": "SUCCESS", "results": results_list, "vehicle_features": vehicle_features, "vehicle_dsp": vehicle_dsp}
