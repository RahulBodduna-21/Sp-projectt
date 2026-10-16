import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from app.core.logging_config import logger
from app.dsp.jerk import compute_adaptive_jerk_threshold
from app.mapping.gps import is_valid_gps

def calculate_anomaly_intensity(
    jerk_peak: float,
    acc_peak: float,
    impact_score: float,
    gyro_response: float
) -> float:
    """Calculates project-specific normalized anomaly intensity score (0-100)."""
    norm_jerk = min(1.0, jerk_peak / 50.0)
    norm_acc = min(1.0, acc_peak / 20.0)
    norm_impact = min(1.0, impact_score / 10.0)
    norm_gyro = min(1.0, gyro_response / 5.0)

    score = 100.0 * (0.35 * norm_jerk + 0.30 * norm_acc + 0.20 * norm_impact + 0.15 * norm_gyro)
    return float(np.clip(round(score, 1), 0.0, 100.0))

def detect_high_jerk_events(
    df_dsp: pd.DataFrame,
    k_multiplier: float = 3.0,
    merge_gap_seconds: float = 1.5,
    max_gps_accuracy_m: float = 50.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Detects signal-supported high-jerk anomaly events and groups nearby samples into single representative events with 0-100 intensity score."""
    if df_dsp is None or df_dsp.empty:
        return pd.DataFrame(), {}

    df = df_dsp.copy()
    jerk_signal = df["jerk"].values if "jerk" in df.columns else np.zeros(len(df))
    abs_jerk = np.abs(jerk_signal)

    jerk_center, jerk_mad, jerk_threshold = compute_adaptive_jerk_threshold(jerk_signal, k_multiplier=k_multiplier)

    high_jerk_mask = abs_jerk >= jerk_threshold
    high_jerk_count = int(high_jerk_mask.sum())

    event_list: List[Dict[str, Any]] = []

    if high_jerk_count > 0:
        event_counter = 1
        for sid, group in df.groupby("session_id"):
            group_sorted = group.sort_values("timestamp").reset_index(drop=True)
            grp_jerk = np.abs(group_sorted["jerk"].values)
            grp_high_mask = grp_jerk >= jerk_threshold

            high_indices = np.where(grp_high_mask)[0]
            if len(high_indices) == 0:
                continue

            # Group consecutive/nearby high-jerk samples within merge_gap_seconds (0.5-2.0s)
            clusters: List[List[int]] = []
            curr_cluster = [high_indices[0]]

            for idx in high_indices[1:]:
                prev_idx = curr_cluster[-1]
                t_prev = group_sorted["timestamp"].iloc[prev_idx]
                t_curr = group_sorted["timestamp"].iloc[idx]
                
                dt = (t_curr - t_prev).total_seconds() if pd.notna(t_prev) and pd.notna(t_curr) else (idx - prev_idx) * 0.02

                if dt <= merge_gap_seconds:
                    curr_cluster.append(idx)
                else:
                    clusters.append(curr_cluster)
                    curr_cluster = [idx]
            if curr_cluster:
                clusters.append(curr_cluster)

            # Build representative event for each cluster
            for cl in clusters:
                cluster_df = group_sorted.iloc[cl]
                start_time = cluster_df["timestamp"].min()
                end_time = cluster_df["timestamp"].max()

                # Event duration in seconds
                evt_dur = (end_time - start_time).total_seconds() if pd.notna(start_time) and pd.notna(end_time) else len(cl) * 0.02

                # Representative peak row inside the cluster
                peak_idx_in_cluster = np.argmax(np.abs(cluster_df["jerk"].values))
                peak_row = cluster_df.iloc[peak_idx_in_cluster]
                event_time = peak_row["timestamp"]

                # GPS association - find nearest valid GPS sample within session
                valid_gps_group = group_sorted.dropna(subset=["latitude", "longitude"])
                valid_gps_group = valid_gps_group[
                    valid_gps_group.apply(lambda r: is_valid_gps(r["latitude"], r["longitude"]), axis=1)
                ]

                if not valid_gps_group.empty:
                    if pd.notna(event_time) and "timestamp" in valid_gps_group.columns:
                        time_diffs = np.abs((valid_gps_group["timestamp"] - event_time).dt.total_seconds())
                        best_gps_idx = time_diffs.idxmin()
                        best_gps_row = valid_gps_group.loc[best_gps_idx]
                    else:
                        best_gps_row = valid_gps_group.iloc[0]

                    lat = float(best_gps_row["latitude"])
                    lon = float(best_gps_row["longitude"])
                    gps_acc = float(best_gps_row.get("gps_accuracy_m", 0.0))
                else:
                    lat = np.nan
                    lon = np.nan
                    gps_acc = 999.0

                jerk_peak = float(np.max(np.abs(cluster_df["jerk"].values)))
                jerk_rms = float(np.sqrt(np.mean(cluster_df["jerk"].values**2)))
                acc_peak = float(np.max(cluster_df["acc_magnitude"].values)) if "acc_magnitude" in cluster_df.columns else 0.0
                impact_score = float(np.max(cluster_df["impact_score"].values)) if "impact_score" in cluster_df.columns else 0.0
                gyro_response = float(np.max(cluster_df["gyro_magnitude"].values)) if "gyro_magnitude" in cluster_df.columns else 0.0

                intensity_score = calculate_anomaly_intensity(jerk_peak, acc_peak, impact_score, gyro_response)

                gt_label = cluster_df["label"].dropna().mode().iloc[0] if len(cluster_df["label"].dropna()) > 0 else "UNKNOWN"
                speed_kmh = float(cluster_df["speed_kmh"].median()) if "speed_kmh" in cluster_df.columns else 0.0

                # Signal-based classification: Distinguish Potholes vs Speed Breakers
                # Speed Breakers cause sustained vehicle pitch/roll elevation (duration >= 2.0s or gyro response > 3.0 rad/s)
                # Potholes cause sharp transient vertical wheel drops (duration < 2.0s)
                if evt_dur >= 2.0 or gyro_response > 3.0:
                    heur_class = "SPEED BREAKER"
                else:
                    heur_class = "POTHOLE"

                gps_valid = is_valid_gps(lat, lon) and (gps_acc <= max_gps_accuracy_m)

                event_list.append({
                    "event_id": f"EVT_{sid}_{event_counter:03d}",
                    "session_id": str(sid),
                    "start_time": start_time,
                    "end_time": end_time,
                    "timestamp": event_time,
                    "latitude": lat,
                    "longitude": lon,
                    "event_duration_s": round(evt_dur, 2),
                    "number_of_high_jerk_samples": len(cl),
                    "jerk_peak": round(jerk_peak, 4),
                    "jerk_rms": round(jerk_rms, 4),
                    "acceleration_peak": round(acc_peak, 4),
                    "impact_score": round(impact_score, 4),
                    "gyroscope_response": round(gyro_response, 4),
                    "intensity_score": intensity_score,
                    "speed_kmh": round(speed_kmh, 1),
                    "gps_accuracy_m": round(gps_acc, 1),
                    "ground_truth_label": gt_label,
                    "heuristic_class": heur_class,
                    "predicted_class": heur_class,
                    "confidence": 0.85,
                    "gps_valid": gps_valid
                })
                event_counter += 1

    df_events = pd.DataFrame(event_list) if event_list else pd.DataFrame()

    avg_intensity = float(df_events["intensity_score"].mean()) if not df_events.empty else 0.0
    max_intensity = float(df_events["intensity_score"].max()) if not df_events.empty else 0.0

    stats = {
        "jerk_median": jerk_center,
        "jerk_mad": jerk_mad,
        "jerk_threshold": jerk_threshold,
        "high_jerk_samples_count": high_jerk_count,
        "grouped_anomaly_events_count": len(df_events),
        "valid_gps_events_count": int(df_events["gps_valid"].sum()) if not df_events.empty else 0,
        "average_anomaly_intensity": round(avg_intensity, 1),
        "maximum_anomaly_intensity": round(max_intensity, 1)
    }

    logger.info(f"Anomaly Detection -> Jerk Median: {jerk_center:.3f}, MAD: {jerk_mad:.3f}, Threshold: {jerk_threshold:.3f}")
    logger.info(f"High-Jerk Samples: {high_jerk_count} -> Grouped Anomaly Events: {len(df_events)}, Avg Intensity: {avg_intensity:.1f}/100")

    return df_events, stats

def annotate_signal_labels(
    df_dsp: pd.DataFrame,
    df_events: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Annotates df_dsp with signal-derived target classes (POTHOLE, SPEED BREAKER, ROUGH, SMOOTH) if raw labels are OTHER or missing."""
    if df_dsp is None or df_dsp.empty:
        return df_dsp

    df = df_dsp.copy()
    if "label" not in df.columns:
        df["label"] = "OTHER"

    # Check existing valid explicit labels
    valid_mask = df["label"].isin(["POTHOLE", "SPEED BREAKER", "SMOOTH", "ROUGH"])
    if valid_mask.sum() > 0.05 * len(df):
        return df

    logger.info("Annotating unlabeled dataset with signal-supported target classes (POTHOLE, SPEED BREAKER, ROUGH, SMOOTH)...")

    # Detect high jerk events if not passed
    if df_events is None or df_events.empty:
        df_events, _ = detect_high_jerk_events(df)

    # Initialize default non-event label to SMOOTH
    annotated_labels = np.array(["SMOOTH"] * len(df), dtype=object)

    # Classify non-event background road roughness using rolling acc std
    acc_col = "acc_magnitude_calibrated" if "acc_magnitude_calibrated" in df.columns else ("acc_magnitude" if "acc_magnitude" in df.columns else "acc_z")
    if acc_col in df.columns:
        roll_std = df[acc_col].rolling(window=100, min_periods=1, center=True).std().fillna(0.0).values
        rough_mask = roll_std > 1.5
        annotated_labels[rough_mask] = "ROUGH"

    # Assign detected anomaly event types
    if df_events is not None and not df_events.empty:
        for _, evt in df_events.iterrows():
            st = evt["start_time"]
            et = evt["end_time"]
            evt_class = evt.get("heuristic_class", evt.get("predicted_class", "POTHOLE"))
            evt_sid = str(evt.get("session_id", ""))
            
            mask = (df["session_id"].astype(str) == evt_sid) & (df["timestamp"] >= st) & (df["timestamp"] <= et)
            if mask.sum() == 0:
                # Fallback to timestamp vicinity
                mask = (df["timestamp"] >= st) & (df["timestamp"] <= et)
            annotated_labels[mask] = evt_class

    df["label"] = annotated_labels
    counts = pd.Series(annotated_labels).value_counts().to_dict()
    logger.info(f"Signal-derived class annotation completed: {counts}")
    return df

