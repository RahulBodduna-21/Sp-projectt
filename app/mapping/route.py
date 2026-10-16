import os
import folium
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from app.core.config import get_project_root
from app.core.logging_config import logger
from app.mapping.gps import filter_valid_gps, is_valid_gps

# Unique RoadSense Anomaly Color System (Rough Road = Brown #8B4513, Smooth Road = Glacier Blue #00A8E8)
COLOR_MAP = {
    "POTHOLE": "#E600A9",            # Fuchsia / Magenta
    "SPEED BREAKER": "#FF7A00",      # Orange
    "ROUGH ROAD ANOMALY": "#8A2BE2", # Violet / Purple
    "OTHER ANOMALY": "#8A2BE2",      # Violet / Purple
    "ROUGH": "#8B4513",              # Brown (Saddle Brown)
    "SMOOTH": "#00A8E8",             # Glacier Blue
    "OTHER": "#8A2BE2"
}

def create_anomaly_icon(anomaly_type: str) -> folium.DivIcon:
    """Creates a glowing custom SVG DivIcon for RoadSense anomaly markers."""
    lbl = str(anomaly_type).upper()
    if "POTHOLE" in lbl:
        color = "#E600A9"
        html = f"""
        <div style="width:26px; height:26px; filter: drop-shadow(0px 0px 6px {color}); pointer-events: auto;">
          <svg width="26" height="26" viewBox="0 0 26 26">
            <circle cx="13" cy="13" r="10" fill="{color}" stroke="#FFFFFF" stroke-width="2"/>
            <circle cx="13" cy="13" r="4" fill="#FFFFFF"/>
          </svg>
        </div>
        """
    elif "SPEED" in lbl or "BREAKER" in lbl:
        color = "#FF7A00"
        html = f"""
        <div style="width:26px; height:26px; filter: drop-shadow(0px 0px 6px {color}); pointer-events: auto;">
          <svg width="26" height="26" viewBox="0 0 26 26">
            <polygon points="13,2 24,13 13,24 2,13" fill="{color}" stroke="#FFFFFF" stroke-width="2"/>
            <polygon points="13,8 18,13 13,18 8,13" fill="#FFFFFF"/>
          </svg>
        </div>
        """
    else:
        color = "#8A2BE2"
        html = f"""
        <div style="width:26px; height:26px; filter: drop-shadow(0px 0px 6px {color}); pointer-events: auto;">
          <svg width="26" height="26" viewBox="0 0 26 26">
            <polygon points="7,2 19,2 24,13 19,24 7,24 2,13" fill="{color}" stroke="#FFFFFF" stroke-width="2"/>
            <polygon points="10,7 16,7 19,13 16,19 10,19 7,13" fill="#FFFFFF"/>
          </svg>
        </div>
        """
    return folium.DivIcon(
        html=html,
        icon_size=(26, 26),
        icon_anchor=(13, 13)
    )

def create_route_badge(label: str, is_start: bool = True) -> folium.DivIcon:
    """Creates custom route Start/End badges avoiding generic red/green map pins."""
    bg_color = "#00A8E8" if is_start else "#4A154B"
    text = f"START ({label})" if is_start else f"END ({label})"
    html = f"""
    <div style="
        background-color: {bg_color};
        color: #ffffff;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 10px;
        font-weight: 800;
        padding: 4px 8px;
        border-radius: 12px;
        border: 2px solid #ffffff;
        box-shadow: 0 0 8px {bg_color};
        white-space: nowrap;
        display: inline-block;
        text-align: center;
    ">
        {text}
    </div>
    """
    return folium.DivIcon(
        html=html,
        icon_size=(70, 20),
        icon_anchor=(35, 10)
    )

def create_popup_html(title: str, header_bg: str, items: List[tuple]) -> str:
    """Creates styled HTML popups with classification header color matching."""
    rows = []
    for k, v in items:
        rows.append(
            f"<tr style='border-bottom: 1px solid #f1f5f9;'>"
            f"<td style='padding: 4px 8px; font-weight: 600; color: #475569; font-size: 11px;'>{k}</td>"
            f"<td style='padding: 4px 8px; color: #0f172a; font-weight: 500; font-size: 11px;'>{v}</td>"
            f"</tr>"
        )
    table_content = "".join(rows)
    return f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 260px; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.15); border: 1px solid #e2e8f0; background: #ffffff;">
        <div style="background-color: {header_bg}; color: #ffffff; padding: 8px 12px; font-weight: 700; font-size: 12px; letter-spacing: 0.5px;">
            {title}
        </div>
        <div style="padding: 6px 4px;">
            <table style="width: 100%; border-collapse: collapse;">
                {table_content}
            </table>
        </div>
    </div>
    """

LEGEND_AND_FILTER_HTML = """
<div style="
    position: fixed; 
    top: 20px; 
    right: 20px; 
    z-index: 9999; 
    background-color: rgba(15, 23, 42, 0.94); 
    backdrop-filter: blur(8px);
    padding: 12px 16px; 
    border-radius: 10px; 
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
    font-size: 12px; 
    color: #f8fafc; 
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); 
    border: 1px solid rgba(255, 255, 255, 0.15);
    min-width: 220px;
">
    <div style="font-weight: 700; font-size: 13px; margin-bottom: 6px; color: #ffffff;">
        Vehicle Type Filter
    </div>
    <select id="vehicleMapFilter" onchange="filterVehicleLayers(this.value)" style="
        width: 100%;
        background-color: #1e293b;
        color: #f8fafc;
        border: 1px solid #475569;
        padding: 6px 10px;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
    ">
        <option value="ALL">ALL VEHICLES</option>
        <option value="2-WHEELER">2-WHEELER (Bike)</option>
        <option value="3-WHEELER">3-WHEELER (Auto)</option>
        <option value="4-WHEELER">4-WHEELER (Car)</option>
    </select>
</div>

<div style="
    position: fixed; 
    bottom: 30px; 
    left: 30px; 
    z-index: 9999; 
    background-color: rgba(15, 23, 42, 0.92); 
    backdrop-filter: blur(8px);
    padding: 14px 18px; 
    border-radius: 10px; 
    font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
    font-size: 12px; 
    color: #f8fafc; 
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5); 
    border: 1px solid rgba(255, 255, 255, 0.15);
    min-width: 240px;
">
    <div style="font-weight: 700; font-size: 14px; margin-bottom: 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.2); padding-bottom: 6px; color: #ffffff;">
        RoadSense AI — Road Anomaly Map
    </div>
    
    <div style="display: flex; align-items: center; margin-bottom: 7px;">
        <svg width="20" height="20" viewBox="0 0 26 26" style="margin-right: 10px; filter: drop-shadow(0px 0px 4px #E600A9);">
            <circle cx="13" cy="13" r="10" fill="#E600A9" stroke="#FFFFFF" stroke-width="2"/>
            <circle cx="13" cy="13" r="4" fill="#FFFFFF"/>
        </svg>
        <span style="font-weight: 500;">Pothole <span style="color: #94a3b8; font-size: 11px;">(#E600A9)</span></span>
    </div>
    
    <div style="display: flex; align-items: center; margin-bottom: 7px;">
        <svg width="20" height="20" viewBox="0 0 26 26" style="margin-right: 10px; filter: drop-shadow(0px 0px 4px #FF7A00);">
            <polygon points="13,2 24,13 13,24 2,13" fill="#FF7A00" stroke="#FFFFFF" stroke-width="2"/>
            <polygon points="13,8 18,13 13,18 8,13" fill="#FFFFFF"/>
        </svg>
        <span style="font-weight: 500;">Speed Breaker <span style="color: #94a3b8; font-size: 11px;">(#FF7A00)</span></span>
    </div>
    
    <div style="display: flex; align-items: center; margin-bottom: 7px;">
        <svg width="20" height="20" viewBox="0 0 26 26" style="margin-right: 10px; filter: drop-shadow(0px 0px 4px #8A2BE2);">
            <polygon points="7,2 19,2 24,13 19,24 7,24 2,13" fill="#8A2BE2" stroke="#FFFFFF" stroke-width="2"/>
            <polygon points="10,7 16,7 19,13 16,19 10,19 7,13" fill="#FFFFFF"/>
        </svg>
        <span style="font-weight: 500;">Other Anomaly <span style="color: #94a3b8; font-size: 11px;">(#8A2BE2)</span></span>
    </div>
    
    <div style="display: flex; align-items: center; margin-bottom: 7px; margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 6px;">
        <span style="display: inline-block; width: 22px; height: 5px; background-color: #00A8E8; margin-right: 10px; border-radius: 2px;"></span>
        <span style="font-weight: 500;">Smooth Road Segment <span style="color: #94a3b8; font-size: 11px;">(Glacier Blue #00A8E8)</span></span>
    </div>
    
    <div style="display: flex; align-items: center;">
        <span style="display: inline-block; width: 22px; height: 0; border-top: 4px dashed #8B4513; margin-right: 10px;"></span>
        <span style="font-weight: 500;">Rough Road Segment <span style="color: #94a3b8; font-size: 11px;">(Brown #8B4513)</span></span>
    </div>
</div>

<script>
function filterVehicleLayers(val) {
    var layers = document.querySelectorAll('.leaflet-interactive, .leaflet-marker-icon');
    // Toggle vehicle feature groups via Leaflet layer control
    var inputs = document.querySelectorAll('.leaflet-control-layers-overlays input[type=checkbox]');
    inputs.forEach(function(inp) {
        var labelText = inp.nextSibling ? inp.nextSibling.textContent.trim().toUpperCase() : '';
        if (val === 'ALL') {
            if (!inp.checked) inp.click();
        } else {
            if (labelText.includes(val)) {
                if (!inp.checked) inp.click();
            } else {
                if (inp.checked) inp.click();
            }
        }
    });
}
</script>
"""

def generate_interactive_map(
    df_raw: pd.DataFrame,
    df_ground_truth: Optional[pd.DataFrame] = None,
    df_anomaly_events: Optional[pd.DataFrame] = None,
    df_segments: Optional[pd.DataFrame] = None,
    df_predictions: Optional[pd.DataFrame] = None,
    output_html_path: Optional[str] = None
) -> folium.Map:
    """Generates dynamic multi-vehicle Leaflet/Folium HTML map with vehicle filtering (2W, 3W, 4W, ALL VEHICLES)."""
    df_valid_gps = filter_valid_gps(df_raw)
    
    if df_valid_gps.empty and (df_predictions is None or filter_valid_gps(df_predictions).empty):
        logger.warning("No valid GPS data found in uploaded dataset. Creating default map.")
        m = folium.Map(location=[0.0, 0.0], zoom_start=2)
        if output_html_path:
            os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
            m.save(output_html_path)
        return m

    # Calculate dynamic bounding box and map center from uploaded GPS data
    all_lats = df_valid_gps["latitude"].tolist()
    all_lons = df_valid_gps["longitude"].tolist()
    
    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lon, max_lon = min(all_lons), max(all_lons)
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")
    
    # Automatically fit map viewport bounds to the actual dataset route
    if min_lat != max_lat or min_lon != max_lon:
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    # Determine vehicle categories present
    if "vehicle_type" in df_valid_gps.columns:
        vehicle_types = df_valid_gps["vehicle_type"].unique().tolist()
    else:
        vehicle_types = ["2-WHEELER"]

    # 1. Feature Groups per Vehicle Type for Road Segments & Routes
    fg_dict = {}
    for v_type in ["2-WHEELER", "3-WHEELER", "4-WHEELER"]:
        fg_dict[v_type] = folium.FeatureGroup(name=f"{v_type} Overlay")

    # Add Road Segments
    if df_segments is not None and not df_segments.empty:
        for _, seg in df_segments.iterrows():
            coords = seg.get("coordinates", [])
            if len(coords) > 1:
                v_type = str(seg.get("vehicle_type", "2-WHEELER")).upper()
                v_key = "3-WHEELER" if "3" in v_type else ("4-WHEELER" if "4" in v_type else "2-WHEELER")
                fg = fg_dict.get(v_key, fg_dict["2-WHEELER"])

                road_cls = str(seg.get("road_classification", "SMOOTH ROAD")).upper()
                is_smooth = "SMOOTH" in road_cls
                color = "#00A8E8" if is_smooth else "#8B4513"
                weight = 6 if is_smooth else 7
                opacity = 0.85 if is_smooth else 0.9
                dash_array = None if is_smooth else "12, 8"

                popup_html = create_popup_html(
                    title=f"[{v_key}] {road_cls}",
                    header_bg=color,
                    items=[
                        ("Vehicle Type", v_key),
                        ("Segment ID", str(seg.get('segment_id'))),
                        ("Duration", f"{seg.get('duration_seconds')} s"),
                        ("Roughness Score", f"{seg.get('roughness_score')}/100"),
                        ("Total Anomalies", str(seg.get('total_anomalies'))),
                        ("Potholes", str(seg.get('potholes'))),
                        ("Speed Breakers", str(seg.get('speed_breakers'))),
                        ("Average Intensity", f"{seg.get('average_intensity')}/100"),
                        ("Average Jerk", f"{seg.get('average_jerk')} m/s³")
                    ]
                )

                folium.PolyLine(
                    coords,
                    color=color,
                    weight=weight,
                    opacity=opacity,
                    dash_array=dash_array,
                    popup=folium.Popup(popup_html, max_width=320)
                ).add_to(fg)
    elif not df_valid_gps.empty:
        for sid, group in df_valid_gps.groupby("session_id"):
            v_type = str(group.get("vehicle_type", pd.Series(["2-WHEELER"])).iloc[0]).upper()
            v_key = "3-WHEELER" if "3" in v_type else ("4-WHEELER" if "4" in v_type else "2-WHEELER")
            fg = fg_dict.get(v_key, fg_dict["2-WHEELER"])

            group_sorted = group.sort_values("timestamp").reset_index(drop=True)
            coords = [[float(row["latitude"]), float(row["longitude"])] for _, row in group_sorted.iterrows()]
            if len(coords) > 1:
                folium.PolyLine(
                    coords,
                    color="#00A8E8",
                    weight=5,
                    opacity=0.85,
                    popup=f"[{v_key}] Session Route: {sid}"
                ).add_to(fg)

    # Add START and END custom badge markers
    if not df_valid_gps.empty:
        for sid, group in df_valid_gps.groupby("session_id"):
            v_type = str(group.get("vehicle_type", pd.Series(["2-WHEELER"])).iloc[0]).upper()
            v_key = "3-WHEELER" if "3" in v_type else ("4-WHEELER" if "4" in v_type else "2-WHEELER")
            fg = fg_dict.get(v_key, fg_dict["2-WHEELER"])

            group_sorted = group.sort_values("timestamp").reset_index(drop=True)
            coords = [[float(row["latitude"]), float(row["longitude"])] for _, row in group_sorted.iterrows()]
            
            start_pt = coords[0]
            start_popup = create_popup_html(
                title=f"[{v_key} ROUTE START]",
                header_bg="#00A8E8",
                items=[
                    ("Vehicle Type", v_key),
                    ("Session ID", str(sid)),
                    ("Time", str(group_sorted['timestamp'].iloc[0])),
                    ("Latitude", f"{start_pt[0]:.6f}"),
                    ("Longitude", f"{start_pt[1]:.6f}")
                ]
            )
            folium.Marker(
                location=start_pt,
                popup=folium.Popup(start_popup, max_width=280),
                icon=create_route_badge(v_key, is_start=True)
            ).add_to(fg)

            end_pt = coords[-1]
            end_popup = create_popup_html(
                title=f"[{v_key} ROUTE END]",
                header_bg="#4A154B",
                items=[
                    ("Vehicle Type", v_key),
                    ("Session ID", str(sid)),
                    ("Time", str(group_sorted['timestamp'].iloc[-1])),
                    ("Latitude", f"{end_pt[0]:.6f}"),
                    ("Longitude", f"{end_pt[1]:.6f}")
                ]
            )
            folium.Marker(
                location=end_pt,
                popup=folium.Popup(end_popup, max_width=280),
                icon=create_route_badge(v_key, is_start=False)
            ).add_to(fg)

    # 2. Feature Group: Detected Anomalies per Vehicle
    if df_anomaly_events is not None and not df_anomaly_events.empty:
        df_evt_valid = filter_valid_gps(df_anomaly_events)
        df_evt_valid = df_evt_valid[df_evt_valid["gps_valid"] == True] if "gps_valid" in df_evt_valid.columns else df_evt_valid
        
        for _, row in df_evt_valid.iterrows():
            v_type = str(row.get("vehicle_type", "2-WHEELER")).upper()
            v_key = "3-WHEELER" if "3" in v_type else ("4-WHEELER" if "4" in v_type else "2-WHEELER")
            fg = fg_dict.get(v_key, fg_dict["2-WHEELER"])

            pred_lbl = str(row.get("predicted_class", row.get("heuristic_class", "OTHER ANOMALY"))).upper()
            color = COLOR_MAP.get(pred_lbl, "#8A2BE2")
            conf = float(row.get("confidence", 0.0))
            j_peak = float(row.get("jerk_peak", 0.0))
            j_rms = float(row.get("jerk_rms", 0.0))
            impact = float(row.get("impact_score", 0.0))
            intensity = float(row.get("intensity_score", 0.0))
            samples_cnt = int(row.get("number_of_high_jerk_samples", 1))
            
            popup_detected = create_popup_html(
                title=f"[{v_key} ANOMALY] {pred_lbl}",
                header_bg=color,
                items=[
                    ("Vehicle Type", v_key),
                    ("Detection", "High Jerk + ML"),
                    ("Intensity", f"{intensity:.1f}/100"),
                    ("Jerk Peak", f"{j_peak:.2f} m/s³"),
                    ("Jerk RMS", f"{j_rms:.2f} m/s³"),
                    ("Impact Score", f"{impact:.3f}"),
                    ("Confidence", f"{conf*100:.1f}%"),
                    ("Speed", f"{row.get('speed_kmh', 0.0):.1f} km/h"),
                    ("Latitude", f"{row.get('latitude'):.6f}"),
                    ("Longitude", f"{row.get('longitude'):.6f}"),
                    ("Session", str(row.get('session_id')))
                ]
            )

            folium.Marker(
                location=[float(row["latitude"]), float(row["longitude"])],
                icon=create_anomaly_icon(pred_lbl),
                popup=folium.Popup(popup_detected, max_width=320)
            ).add_to(fg)

    # Add all vehicle feature groups to map
    for v_key in ["2-WHEELER", "3-WHEELER", "4-WHEELER"]:
        fg_dict[v_key].add_to(m)

    folium.LayerControl().add_to(m)
    
    # Add custom legend and vehicle filter HTML overlay
    m.get_root().html.add_child(folium.Element(LEGEND_AND_FILTER_HTML))

    if output_html_path:
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        m.save(output_html_path)
        logger.info(f"Saved multi-vehicle interactive HTML map to {output_html_path}")

    return m
