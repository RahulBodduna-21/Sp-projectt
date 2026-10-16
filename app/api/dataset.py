import io
import uuid
import pandas as pd
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from app.dsp.validation import validate_dataset

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])

# In-memory session data store
ANALYSIS_STORE: Dict[str, Dict[str, Any]] = {}

class DatasetSummaryResponse(BaseModel):
    analysis_id: str
    total_rows: int
    valid_rows: int
    invalid_rows: int
    number_of_sessions: int
    total_duration_seconds: float
    estimated_sampling_rate_hz: float
    label_distribution: Dict[str, int]
    gps_coverage: Dict[str, Any]

@router.post("/upload", response_model=DatasetSummaryResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads a smartphone CSV dataset, assigns an analysis_id, and validates it."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df_raw = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {e}")

    try:
        df_valid, report = validate_dataset(df_raw)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Dataset validation failed: {e}")

    analysis_id = str(uuid.uuid4())[:8]
    ANALYSIS_STORE[analysis_id] = {
        "df_raw": df_raw,
        "df_valid": df_valid,
        "validation_report": report
    }

    return DatasetSummaryResponse(analysis_id=analysis_id, **report)

@router.get("/summary", response_model=DatasetSummaryResponse)
async def get_dataset_summary(analysis_id: str):
    """Retrieves validation summary for a given analysis_id."""
    if analysis_id not in ANALYSIS_STORE:
        raise HTTPException(status_code=404, detail="Analysis ID not found. Upload dataset first.")
    
    report = ANALYSIS_STORE[analysis_id]["validation_report"]
    return DatasetSummaryResponse(analysis_id=analysis_id, **report)
