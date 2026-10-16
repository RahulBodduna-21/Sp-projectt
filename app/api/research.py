import os
import pandas as pd
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from app.api.dataset import ANALYSIS_STORE
from app.core.config import get_project_root
from app.research.ablation import run_ablation_study
from app.research.cross_vehicle import run_cross_vehicle_study

router = APIRouter(prefix="/api/research", tags=["Research Experiments"])

@router.post("/ablation")
async def trigger_ablation(analysis_id: str):
    """Triggers automated 5-stage ablation study on uploaded dataset."""
    if analysis_id not in ANALYSIS_STORE:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")

    df_raw = ANALYSIS_STORE[analysis_id]["df_raw"]
    try:
        df_results, results_list = run_ablation_study(df_raw)
        ANALYSIS_STORE[analysis_id]["ablation_results"] = results_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ablation study failed: {e}")

    return {
        "analysis_id": analysis_id,
        "ablation_stages_count": len(results_list),
        "results": results_list
    }

@router.post("/cross-vehicle")
async def trigger_cross_vehicle(analysis_id: str):
    """Triggers Leave-One-Vehicle-Out cross-vehicle experiment."""
    if analysis_id not in ANALYSIS_STORE:
        raise HTTPException(status_code=404, detail="Analysis ID not found.")

    df_raw = ANALYSIS_STORE[analysis_id]["df_raw"]
    df_results, res_dict = run_cross_vehicle_study({"Upload": df_raw})
    return res_dict

@router.get("/results")
async def get_research_results(analysis_id: str):
    """Returns saved research tables and figure paths."""
    res_dir = os.path.join(get_project_root(), "results")
    abl_path = os.path.join(res_dir, "ablation_results.csv")
    cv_path = os.path.join(res_dir, "cross_vehicle_results.csv")

    ablation_data = pd.read_csv(abl_path).to_dict(orient="records") if os.path.exists(abl_path) else []
    cross_data = pd.read_csv(cv_path).to_dict(orient="records") if os.path.exists(cv_path) else []

    fig_dir = os.path.join(res_dir, "figures")
    figures = [os.path.basename(f) for f in os.listdir(fig_dir)] if os.path.exists(fig_dir) else []

    return {
        "analysis_id": analysis_id,
        "ablation_results": ablation_data,
        "cross_vehicle_results": cross_data,
        "generated_figures": figures
    }
