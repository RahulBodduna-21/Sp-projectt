import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from app.core.logging_config import logger
from app.mapping.gps import filter_valid_gps

def calculate_haversine_distance_km(coords: List[Tuple[float, float]]) -> float:
    """Calculates total distance in km for a list of (lat, lon) coordinates using Haversine formula."""
    if len(coords) < 2:
        return 0.0

    lats = np.radians([c[0] for c in coords])
    lons = np.radians([c[1] for c in coords])

    dlats = np.diff(lats)
    dlons = np.diff(lons)

    a = np.sin(dlats / 2.0)**2 + np.cos(lats[:-1]) * np.cos(lats[1:]) * np.sin(dlons / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    r_km = 6371.0
    return float(np.sum(r_km * c))

def classify_road_segments(
    df_dsp: pd.DataFrame,
    df_events: Optional[pd.DataFrame] = None,
    segment_duration_seconds: float = 15.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Divides the route into road segments, computes segment statistics, and classifies each segment as SMOOTH ROAD or ROUGH ROAD."""
    if df_dsp is None or df_dsp.empty:
        return pd.DataFrame(), {}

    segments_list: List[Dict[str, Any]] = []

    total_distance_km = 0.0

    for sid, group in df_dsp.groupby("session_id"):
        group_sorted = group.sort_values("timestamp").reset_index(drop=True)
        if len(group_sorted) < 5:
            continue

        start_time = group_sorted["timestamp"].min()
        end_time = group_sorted["timestamp"].max()

        if pd.isna(start_time) or pd.isna(end_time):
            continue

        total_sec = (end_time - start_time).total_seconds()
        if total_sec <= 0:
            total_sec = len(group_sorted) * 0.02

        num_segments = max(1, int(np.ceil(total_sec / segment_duration_seconds)))
        
        # Determine time window boundaries per segment
        t_starts = [start_time + pd.Timedelta(seconds=i * segment_duration_seconds) for i in range(num_segments)]
        
        for i in range(num_segments):
            seg_t_start = t_starts[i]
            seg_t_end = start_time + pd.Timedelta(seconds=(i + 1) * segment_duration_seconds)

            sub_df = group_sorted[(group_sorted["timestamp"] >= seg_t_start) & (group_sorted["timestamp"] < seg_t_end)]
            if sub_df.empty:
                continue

            seg_duration = (sub_df["timestamp"].max() - sub_df["timestamp"].min()).total_seconds()
            if seg_duration <= 0:
                seg_duration = len(sub_df) * 0.02

            # Sub-segment coordinates
            valid_gps_sub = filter_valid_gps(sub_df)
            coords = [[float(r["latitude"]), float(r["longitude"])] for _, r in valid_gps_sub.iterrows()]
            seg_dist_km = calculate_haversine_distance_km([(c[0], c[1]) for c in coords]) if len(coords) > 1 else 0.0
            total_distance_km += seg_dist_km

            # Match events occurring within this road segment time window
            if df_events is not None and not df_events.empty:
                evts_in_seg = df_events[
                    (df_events["session_id"].astype(str) == str(sid)) &
                    (df_events["timestamp"] >= seg_t_start) &
                    (df_events["timestamp"] < seg_t_end)
                ]
            else:
                evts_in_seg = pd.DataFrame()

            total_anomalies = len(evts_in_seg)
            pothole_count = int((evts_in_seg["predicted_class"] == "POTHOLE").sum()) if not evts_in_seg.empty and "predicted_class" in evts_in_seg.columns else 0
            speed_breaker_count = int((evts_in_seg["predicted_class"] == "SPEED BREAKER").sum()) if not evts_in_seg.empty and "predicted_class" in evts_in_seg.columns else 0
            other_count = total_anomalies - (pothole_count + speed_breaker_count)

            density_per_min = (total_anomalies / seg_duration) * 60.0 if seg_duration > 0 else 0.0

            avg_intensity = float(evts_in_seg["intensity_score"].mean()) if not evts_in_seg.empty and "intensity_score" in evts_in_seg.columns else 0.0
            max_intensity = float(evts_in_seg["intensity_score"].max()) if not evts_in_seg.empty and "intensity_score" in evts_in_seg.columns else 0.0

            # Sensor signal statistics
            jerk_vals = sub_df["jerk"].values if "jerk" in sub_df.columns else np.zeros(len(sub_df))
            acc_mag = sub_df["acc_magnitude"].values if "acc_magnitude" in sub_df.columns else np.zeros(len(sub_df))
            gyro_mag = sub_df["gyro_magnitude"].values if "gyro_magnitude" in sub_df.columns else np.zeros(len(sub_df))

            avg_jerk = float(np.mean(np.abs(jerk_vals)))
            jerk_var = float(np.std(jerk_vals))
            acc_var = float(np.std(acc_mag))
            gyro_var = float(np.std(gyro_mag))
            avg_speed = float(sub_df["speed_kmh"].mean()) if "speed_kmh" in sub_df.columns else 0.0

            # Adaptive Roughness Score (0-100)
            roughness_score = 0.40 * min(100.0, density_per_min * 10.0) + 0.30 * avg_intensity + 0.30 * min(100.0, jerk_var * 25.0)
            roughness_score = float(np.clip(roughness_score, 0.0, 100.0))

            # Segment Classification
            if roughness_score >= 35.0 or total_anomalies >= 2:
                road_class = "ROUGH ROAD"
            else:
                road_class = "SMOOTH ROAD"

            segments_list.append({
                "segment_id": f"SEG_{sid}_{i+1:03d}",
                "session_id": str(sid),
                "start_time": seg_t_start,
                "end_time": seg_t_end,
                "duration_seconds": round(seg_duration, 1),
                "distance_km": round(seg_dist_km, 3),
                "road_classification": road_class,
                "roughness_score": round(roughness_score, 1),
                "total_anomalies": total_anomalies,
                "potholes": pothole_count,
                "speed_breakers": speed_breaker_count,
                "other_anomalies": other_count,
                "anomaly_density_per_min": round(density_per_min, 1),
                "average_intensity": round(avg_intensity, 1),
                "maximum_intensity": round(max_intensity, 1),
                "average_jerk": round(avg_jerk, 3),
                "jerk_variability": round(jerk_var, 3),
                "acceleration_variability": round(acc_var, 3),
                "gyroscope_variability": round(gyro_var, 3),
                "average_speed_kmh": round(avg_speed, 1),
                "coordinates": coords
            })

    df_segments = pd.DataFrame(segments_list) if segments_list else pd.DataFrame()

    total_segments = len(df_segments)
    smooth_count = int((df_segments["road_classification"] == "SMOOTH ROAD").sum()) if not df_segments.empty else 0
    rough_count = int((df_segments["road_classification"] == "ROUGH ROAD").sum()) if not df_segments.empty else 0

    smooth_pct = round((smooth_count / total_segments * 100.0), 1) if total_segments > 0 else 0.0
    rough_pct = round((rough_count / total_segments * 100.0), 1) if total_segments > 0 else 0.0

    summary_stats = {
        "total_road_distance_km": round(total_distance_km, 2),
        "total_road_segments": total_segments,
        "smooth_road_segments": smooth_count,
        "rough_road_segments": rough_count,
        "smooth_road_percentage": smooth_pct,
        "rough_road_percentage": rough_pct
    }

    logger.info(f"Road Segmentation -> Total Segments: {total_segments}, Smooth: {smooth_count} ({smooth_pct}%), Rough: {rough_count} ({rough_pct}%)")
    return df_segments, summary_stats
