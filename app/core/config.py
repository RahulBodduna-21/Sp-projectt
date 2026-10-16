import os
import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional

class SamplingConfig(BaseModel):
    target_hz: float = 50.0
    regularity_threshold: float = 0.1

class GravityConfig(BaseModel):
    cutoff_hz: float = 0.5
    order: int = 3

class ButterworthConfig(BaseModel):
    order: int = 4
    low_cutoff_hz: float = 0.7
    high_cutoff_hz: float = 15.0

class HampelConfig(BaseModel):
    window_size: int = 7
    n_sigma: float = 3.0

class WindowConfig(BaseModel):
    duration_seconds: float = 2.0
    overlap_seconds: float = 0.0

class ModelConfig(BaseModel):
    type: str = "random_forest"
    n_estimators: int = 300
    random_state: int = 42
    class_weight: str = "balanced"
    n_jobs: int = -1

class MappingConfig(BaseModel):
    gps_association_method: str = "midpoint"
    min_gps_accuracy_m: float = 50.0

class PathsConfig(BaseModel):
    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    data_features: str = "data/features"
    data_predictions: str = "data/predictions"
    models: str = "models"
    results: str = "results"

class AppConfig(BaseModel):
    sampling: SamplingConfig = SamplingConfig()
    gravity: GravityConfig = GravityConfig()
    butterworth: ButterworthConfig = ButterworthConfig()
    hampel: HampelConfig = HampelConfig()
    window: WindowConfig = WindowConfig()
    model: ModelConfig = ModelConfig()
    classes: List[str] = ["SMOOTH", "ROUGH", "POTHOLE", "SPEED BREAKER"]
    mapping: MappingConfig = MappingConfig()
    paths: PathsConfig = PathsConfig()

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def load_config(config_path: Optional[str] = None) -> AppConfig:
    root = get_project_root()
    if config_path is None:
        config_path = os.path.join(root, "config.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data:
                return AppConfig(**data)
    return AppConfig()
