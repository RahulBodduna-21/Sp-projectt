from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from typing import Dict, Any, List, Optional
from app.api.dataset import ANALYSIS_STORE
from app.mapping.ground_truth import extract_ground_truth_anomalies
from app.mapping.route import generate_interactive_map

router = APIRouter(prefix="/api/map", tags=["Mapping"])

@router.get("/route")
async def get_map_route(analysis_id: str):
    """Returns chronological GPS route points per session."""
    if analysis_id not in ANALYSIS_STORE:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")

    df_raw = ANALYSIS_STORE[analysis_id]["df_raw"]
    routes = []
    
    if "latitude" in df_raw.columns and "longitude" in df_raw.columns:
        valid_gps = df_raw.dropna(subset=["latitude", "longitude"])
        valid_gps = valid_gps[(valid_gps["latitude"] != 0) & (valid_gps["longitude"] != 0)]
        for sid, group in valid_gps.groupby("session_id"):
            coords = group[["latitude", "longitude", "speed_kmh", "timestamp"]].to_dict(orient="records")
            routes.append({"session_id": sid, "point_count": len(coords), "coordinates": coords})

    return {"analysis_id": analysis_id, "routes": routes}

@router.get("/ground-truth")
async def get_map_ground_truth(analysis_id: str):
    """Returns independent ground-truth anomaly events parsed from CSV."""
    if analysis_id not in ANALYSIS_STORE:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")

    df_raw = ANALYSIS_STORE[analysis_id]["df_raw"]
    df_gt, stats = extract_ground_truth_anomalies(df_raw)

    return {
        "analysis_id": analysis_id,
        "stats": stats,
        "ground_truth_anomalies": df_gt.to_dict(orient="records") if not df_gt.empty else []
    }

@router.get("/predictions")
async def get_map_predictions(analysis_id: str):
    """Returns predictions associated with window GPS midpoint coordinates."""
    if analysis_id not in ANALYSIS_STORE or "df_predictions" not in ANALYSIS_STORE[analysis_id]:
        raise HTTPException(status_code=404, detail="Predictions not found for map.")

    df_preds = ANALYSIS_STORE[analysis_id]["df_predictions"]
    valid_preds = df_preds.dropna(subset=["latitude", "longitude"])

    return {
        "analysis_id": analysis_id,
        "count": len(valid_preds),
        "predictions": valid_preds.to_dict(orient="records")
    }

@router.get("/html", response_class=HTMLResponse)
async def get_map_html(analysis_id: str):
    """Generates and returns interactive HTML map page."""
    if analysis_id not in ANALYSIS_STORE:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")

    store = ANALYSIS_STORE[analysis_id]
    df_raw = store["df_raw"]
    df_gt, _ = extract_ground_truth_anomalies(df_raw)
    df_preds = store.get("df_predictions")

    m = generate_interactive_map(df_raw, df_ground_truth=df_gt, df_predictions=df_preds)
    return m._repr_html_()
