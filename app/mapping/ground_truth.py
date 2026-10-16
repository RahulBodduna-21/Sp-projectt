import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from app.core.logging_config import logger
from app.dsp.validation import normalize_label
from app.mapping.gps import is_valid_gps

def extract_ground_truth_anomalies(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Extracts ground-truth anomaly events (POTHOLE, SPEED BREAKER) with valid GPS directly from raw CSV."""
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(), {}

    df = df_raw.copy()
    if "label" in df.columns:
        df["label_norm"] = df["label"].apply(normalize_label)
    else:
        df["label_norm"] = None

    total_rows = len(df)
    pothole_mask = df["label_norm"] == "POTHOLE"
    speed_breaker_mask = df["label_norm"] == "SPEED BREAKER"
    
    pothole_rows = int(pothole_mask.sum())
    speed_breaker_rows = int(speed_breaker_mask.sum())

    valid_gps_mask = df.apply(lambda r: is_valid_gps(r.get("latitude"), r.get("longitude")), axis=1)

    potholes_valid_gps = int((pothole_mask & valid_gps_mask).sum())
    speed_breakers_valid_gps = int((speed_breaker_mask & valid_gps_mask).sum())

    # Extract anomaly rows with valid GPS
    anomaly_mask = (pothole_mask | speed_breaker_mask) & valid_gps_mask
    df_anomalies = df[anomaly_mask].copy()

    # Deduplicate consecutive samples of the same anomaly event within 3 seconds/window per session
    dedup_rows: List[pd.Series] = []
    if len(df_anomalies) > 0:
        for (sid, lbl), group in df_anomalies.groupby(["session_id", "label_norm"]):
            group_sorted = group.sort_values("timestamp")
            prev_time = None
            for _, row in group_sorted.iterrows():
                curr_time = pd.to_datetime(row["timestamp"])
                if prev_time is None or (curr_time - prev_time).total_seconds() > 3.0:
                    dedup_rows.append(row)
                    prev_time = curr_time

    df_gt_clean = pd.DataFrame(dedup_rows) if dedup_rows else pd.DataFrame()

    if not df_gt_clean.empty:
        df_gt_clean["label"] = df_gt_clean["label_norm"]
        cols = ["id", "session_id", "timestamp", "label", "latitude", "longitude", "speed_kmh", "gps_accuracy_m"]
        existing_cols = [c for c in cols if c in df_gt_clean.columns]
        df_gt_clean = df_gt_clean[existing_cols].reset_index(drop=True)

    # Detailed debugging logging as required by research protocol
    logger.info("================ GROUND-TRUTH GPS LOGGING ================")
    logger.info(f"Total CSV rows: {total_rows}")
    logger.info(f"Pothole rows: {pothole_rows}")
    logger.info(f"Pothole rows with valid GPS: {potholes_valid_gps}")
    logger.info(f"Speed breaker rows: {speed_breaker_rows}")
    logger.info(f"Speed breaker rows with valid GPS: {speed_breakers_valid_gps}")
    logger.info(f"Ground-truth anomaly events extracted: {len(df_gt_clean)}")

    if not df_gt_clean.empty:
        sample_potholes = df_gt_clean[df_gt_clean["label"] == "POTHOLE"].head(2)
        sample_sb = df_gt_clean[df_gt_clean["label"] == "SPEED BREAKER"].head(2)
        
        if not sample_potholes.empty:
            for _, r in sample_potholes.iterrows():
                logger.info(f"REAL POTHOLE RECORD -> lat: {r.get('latitude')}, lon: {r.get('longitude')}, ts: {r.get('timestamp')}, session: {r.get('session_id')}, id: {r.get('id')}")
        if not sample_sb.empty:
            for _, r in sample_sb.iterrows():
                logger.info(f"REAL SPEED BREAKER RECORD -> lat: {r.get('latitude')}, lon: {r.get('longitude')}, ts: {r.get('timestamp')}, session: {r.get('session_id')}, id: {r.get('id')}")
    logger.info("==========================================================")

    stats = {
        "total_csv_rows": total_rows,
        "pothole_rows": pothole_rows,
        "pothole_rows_valid_gps": potholes_valid_gps,
        "speed_breaker_rows": speed_breaker_rows,
        "speed_breaker_rows_valid_gps": speed_breakers_valid_gps,
        "ground_truth_anomalies_count": len(df_gt_clean)
    }

    return df_gt_clean, stats
