from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.api.dataset import ANALYSIS_STORE
from app.dsp.pipeline import process_dsp_pipeline

router = APIRouter(prefix="/api/dsp", tags=["DSP"])

class DSPProcessResponse(BaseModel):
    analysis_id: str
    status: str
    processed_rows: int
    dsp_summary: Dict[str, Any]

@router.post("/process", response_model=DSPProcessResponse)
async def process_dsp(analysis_id: str):
    """Executes the complete DSP pipeline on the uploaded dataset."""
    if analysis_id not in ANALYSIS_STORE:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")

    store = ANALYSIS_STORE[analysis_id]
    df_raw = store["df_raw"]

    try:
        df_dsp, summary = process_dsp_pipeline(df_raw)
        store["df_dsp"] = df_dsp
        store["dsp_summary"] = summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DSP processing error: {e}")

    return DSPProcessResponse(
        analysis_id=analysis_id,
        status="SUCCESS",
        processed_rows=len(df_dsp),
        dsp_summary=summary
    )

@router.get("/summary")
async def get_dsp_summary(analysis_id: str):
    """Returns DSP execution configuration and metrics summary."""
    if analysis_id not in ANALYSIS_STORE or "dsp_summary" not in ANALYSIS_STORE[analysis_id]:
        raise HTTPException(status_code=404, detail="DSP summary not found for analysis_id.")
    return ANALYSIS_STORE[analysis_id]["dsp_summary"]

@router.get("/signals")
async def get_dsp_signals(analysis_id: str, limit: int = Query(500, ge=10, le=5000)):
    """Returns time-series signal streams for raw, filtered, calibrated, and impact scores."""
    if analysis_id not in ANALYSIS_STORE or "df_dsp" not in ANALYSIS_STORE[analysis_id]:
        raise HTTPException(status_code=404, detail="DSP signals not found. Run DSP process first.")

    df_dsp = ANALYSIS_STORE[analysis_id]["df_dsp"]
    df_sub = df_dsp.iloc[:limit]

    cols = ["elapsed_time_s", "acc_x", "acc_y", "acc_z", "acc_magnitude", "acc_magnitude_calibrated", "jerk", "impact_score", "gyro_magnitude"]
    existing_cols = [c for c in cols if c in df_sub.columns]

    return {
        "analysis_id": analysis_id,
        "sample_count": len(df_sub),
        "signals": df_sub[existing_cols].to_dict(orient="records")
    }
