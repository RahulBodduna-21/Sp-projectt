from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.api.dataset import ANALYSIS_STORE
from app.ml.dataset_builder import build_ml_dataset
from app.ml.evaluator import evaluate_model
from app.ml.trainer import train_classifier
from app.ml.model_registry import save_model_artifact

router = APIRouter(prefix="/api/ml", tags=["Machine Learning"])

class MLTrainResponse(BaseModel):
    analysis_id: str
    status: str
    metrics: Dict[str, Any]

@router.post("/train", response_model=MLTrainResponse)
async def train_ml_model(analysis_id: str):
    """Trains Random Forest model and evaluates performance with session-aware validation."""
    if analysis_id not in ANALYSIS_STORE or "df_features" not in ANALYSIS_STORE[analysis_id]:
        raise HTTPException(status_code=404, detail="Features not found. Run feature extraction first.")

    store = ANALYSIS_STORE[analysis_id]
    df_features = store["df_features"]

    try:
        X, y, groups, le, feat_names, df_valid = build_ml_dataset(df_features)
        metrics, y_test, y_pred = evaluate_model(X, y, groups, le)
        model = train_classifier(X, y)
        
        filepath = save_model_artifact(model, le, feat_names, metadata={"metrics": metrics})
        
        store["model"] = model
        store["label_encoder"] = le
        store["feature_names"] = feat_names
        store["metrics"] = metrics
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"ML training failed: {e}")

    return MLTrainResponse(
        analysis_id=analysis_id,
        status="SUCCESS",
        metrics=metrics
    )

@router.get("/metrics")
async def get_ml_metrics(analysis_id: str):
    """Returns classification evaluation metrics."""
    if analysis_id not in ANALYSIS_STORE or "metrics" not in ANALYSIS_STORE[analysis_id]:
        raise HTTPException(status_code=404, detail="ML metrics not found. Train model first.")

    return ANALYSIS_STORE[analysis_id]["metrics"]
