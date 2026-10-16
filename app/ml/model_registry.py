import os
import joblib
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from app.core.config import AppConfig, get_project_root
from app.core.logging_config import logger

def save_model_artifact(
    model: RandomForestClassifier,
    label_encoder: LabelEncoder,
    feature_names: list,
    metadata: Dict[str, Any],
    filename: str = "random_forest_model.joblib",
    target_dir: Optional[str] = None
) -> str:
    """Saves model, encoder, feature names, and metadata to disk."""
    if target_dir is None:
        target_dir = os.path.join(get_project_root(), "models")
        
    os.makedirs(target_dir, exist_ok=True)
    filepath = os.path.join(target_dir, filename)

    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "metadata": metadata
    }

    joblib.dump(artifact, filepath)
    logger.info(f"Saved model artifact to {filepath}")
    return filepath

def load_model_artifact(filepath: str) -> Tuple[RandomForestClassifier, LabelEncoder, list, Dict[str, Any]]:
    """Loads saved model artifact from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model artifact file not found: {filepath}")

    artifact = joblib.load(filepath)
    logger.info(f"Loaded model artifact from {filepath}")
    return artifact["model"], artifact["label_encoder"], artifact["feature_names"], artifact["metadata"]
