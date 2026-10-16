import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Optional
from app.core.config import AppConfig, load_config
from app.core.exceptions import MLError
from app.core.logging_config import logger

def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Optional[AppConfig] = None
) -> RandomForestClassifier:
    """Trains a Random Forest classifier on training data."""
    if config is None:
        config = load_config()

    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise MLError(f"Cannot train classifier with fewer than 2 classes. Found: {unique_classes}")

    clf = RandomForestClassifier(
        n_estimators=config.model.n_estimators,
        random_state=config.model.random_state,
        class_weight=config.model.class_weight,
        n_jobs=config.model.n_jobs
    )
    
    logger.info(f"Training Random Forest (n_estimators={config.model.n_estimators}) on {X_train.shape[0]} samples...")
    clf.fit(X_train, y_train)
    return clf
