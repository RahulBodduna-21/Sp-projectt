from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from app.api.dataset import ANALYSIS_STORE
from app.ml.predictor import generate_predictions

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])

@router.post("/predict")
async def run_predictions(analysis_id: str):
    """Generates batch predictions on feature windows using trained classifier."""
    if analysis_id not in ANALYSIS_STORE or "model" not in ANALYSIS_STORE[analysis_id]:
        raise HTTPException(status_code=404, detail="Trained model not found for analysis_id. Run ML train first.")

    store = ANALYSIS_STORE[analysis_id]
    model = store["model"]
    le = store["label_encoder"]
    df_features = store["df_features"]
    feat_names = store["feature_names"]

    try:
        df_preds = generate_predictions(model, le, df_features, feat_names)
        store["df_predictions"] = df_preds
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {
        "analysis_id": analysis_id,
        "prediction_count": len(df_preds),
        "predicted_class_counts": df_preds["predicted_class"].value_counts().to_dict(),
        "mean_confidence": float(df_preds["confidence"].mean())
    }

@router.get("")
async def get_predictions_data(analysis_id: str, limit: int = 500):
    """Returns predictions table records."""
    if analysis_id not in ANALYSIS_STORE or "df_predictions" not in ANALYSIS_STORE[analysis_id]:
        raise HTTPException(status_code=404, detail="Predictions not found. Run predictions first.")

    df_preds = ANALYSIS_STORE[analysis_id]["df_predictions"]
    df_sub = df_preds.iloc[:limit]

    return {
        "analysis_id": analysis_id,
        "count": len(df_sub),
        "predictions": df_sub.to_dict(orient="records")
    }
