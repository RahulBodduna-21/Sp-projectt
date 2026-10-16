from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.api.dataset import ANALYSIS_STORE
from app.features.extractor import extract_features

router = APIRouter(prefix="/api/features", tags=["Features"])

class FeatureExtractResponse(BaseModel):
    analysis_id: str
    status: str
    num_windows: int
    num_features: int

@router.post("/extract", response_model=FeatureExtractResponse)
async def extract_dataset_features(analysis_id: str, stage: str = Query("E", pattern="^[A-Ea-e]$")):
    """Extracts windowed features from processed DSP dataset."""
    if analysis_id not in ANALYSIS_STORE or "df_dsp" not in ANALYSIS_STORE[analysis_id]:
        raise HTTPException(status_code=404, detail="DSP signals not found. Run DSP process first.")

    store = ANALYSIS_STORE[analysis_id]
    df_dsp = store["df_dsp"]

    try:
        df_features = extract_features(df_dsp, config_stage=stage.upper())
        store["df_features"] = df_features
        store["feature_stage"] = stage.upper()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction error: {e}")

    meta_cols = ["session_id", "window_start", "window_end", "ground_truth_label", "latitude", "longitude", "speed_kmh", "vehicle_type"]
    feat_count = len([c for c in df_features.columns if c not in meta_cols])

    return FeatureExtractResponse(
        analysis_id=analysis_id,
        status="SUCCESS",
        num_windows=len(df_features),
        num_features=feat_count
    )

@router.get("/summary")
async def get_features_summary(analysis_id: str):
    """Returns summary of extracted feature vectors."""
    if analysis_id not in ANALYSIS_STORE or "df_features" not in ANALYSIS_STORE[analysis_id]:
        raise HTTPException(status_code=404, detail="Features not found. Run feature extraction first.")

    df_feat = ANALYSIS_STORE[analysis_id]["df_features"]
    return {
        "analysis_id": analysis_id,
        "num_windows": len(df_feat),
        "columns": list(df_feat.columns),
        "label_counts": df_feat["ground_truth_label"].value_counts().to_dict()
    }

@router.get("/fft")
async def get_fft_summary(analysis_id: str):
    """Returns frequency-domain FFT spectral metrics across windows."""
    if analysis_id not in ANALYSIS_STORE or "df_features" not in ANALYSIS_STORE[analysis_id]:
        raise HTTPException(status_code=404, detail="Features not found. Run feature extraction first.")

    df_feat = ANALYSIS_STORE[analysis_id]["df_features"]
    fft_cols = [c for c in df_feat.columns if "fft" in c or "freq" in c or "energy" in c or "centroid" in c]

    return {
        "analysis_id": analysis_id,
        "fft_feature_columns": fft_cols,
        "summary": df_feat[fft_cols].describe().to_dict() if fft_cols else {}
    }
