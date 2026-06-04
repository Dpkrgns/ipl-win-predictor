class IPLPredictorError(Exception):
    """Base exception for project-specific failures."""


class DataValidationError(IPLPredictorError):
    """Raised when expected dataset columns or values are missing."""


class ModelArtifactError(IPLPredictorError):
    """Raised when a saved model artifact cannot be loaded or used."""
