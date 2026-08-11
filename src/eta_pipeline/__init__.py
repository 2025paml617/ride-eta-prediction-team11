"""Data pipeline for the NYC taxi ride-ETA project.

Ingest raw trip data, validate it, engineer features, and write a versioned,
model-ready dataset with a manifest describing exactly how it was produced.
"""

__version__ = "1.0.0"

from .config import PipelineConfig

__all__ = ["PipelineConfig", "__version__"]
