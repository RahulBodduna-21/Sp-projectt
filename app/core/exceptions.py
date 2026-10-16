class RoadSenseError(Exception):
    """Base exception for RoadSense AI system."""
    pass

class DatasetValidationError(RoadSenseError):
    """Raised when CSV validation fails."""
    pass

class DSPProcessingError(RoadSenseError):
    """Raised when signal processing error occurs."""
    pass

class FeatureExtractionError(RoadSenseError):
    """Raised when feature extraction fails."""
    pass

class MLError(RoadSenseError):
    """Raised when ML training or prediction fails."""
    pass

class InsufficientDataError(RoadSenseError):
    """Raised when insufficient data exists for an experiment or model training."""
    pass
