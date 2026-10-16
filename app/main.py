import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import dataset, dsp, features, ml, predictions, map, research
from app.core.config import get_project_root

app = FastAPI(
    title="RoadSense AI — Real-World Research Prototype API",
    description="Vehicle-Independent Road Anomaly Classification Through DSP-Based Signal Calibration",
    version="1.0.0"
)

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(dataset.router)
app.include_router(dsp.router)
app.include_router(features.router)
app.include_router(ml.router)
app.include_router(predictions.router)
app.include_router(map.router)
app.include_router(research.router)

# Mount static files for results/figures and results/map
results_path = os.path.join(get_project_root(), "results")
if os.path.exists(results_path):
    app.mount("/results", StaticFiles(directory=results_path), name="results")

@app.get("/")
async def root():
    return {
        "project": "RoadSense AI",
        "description": "DSP-Based Signal Calibration Across Diverse Vehicle and Road Conditions",
        "status": "ONLINE",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
