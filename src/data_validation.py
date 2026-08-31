"""Compatibility wrapper for the canonical data validator."""

from data.data_validation import (
    DataValidationError,
    GPS_COLUMNS,
    REQUIRED_COLUMNS,
    ValidationResult,
    load_and_validate_data,
    validate_ingested_data,
)

__all__ = [
    "DataValidationError", "GPS_COLUMNS", "REQUIRED_COLUMNS", "ValidationResult",
    "load_and_validate_data", "validate_ingested_data",
]
