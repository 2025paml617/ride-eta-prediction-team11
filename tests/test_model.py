"""Unit tests for the ETA model training and pre-processing pipeline."""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.eta_model.train import create_preprocessor, ALL_FEATURES, TARGET
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


class TestModelTraining(unittest.TestCase):
    """Tests for the ML training setup, pre-processing, and inference."""

    def test_preprocessor_structure(self):
        """Test that the preprocessor has expected column mapping."""
        preprocessor = create_preprocessor()
        transformers = preprocessor.transformers
        
        # Verify continuous and passthrough groups
        num_trans = [t for t in transformers if t[0] == "num"]
        pass_trans = [t for t in transformers if t[0] == "passthrough"]
        
        self.assertEqual(len(num_trans), 1)
        self.assertEqual(len(pass_trans), 1)

    def test_end_to_end_pipeline_and_serialization(self):
        """Test that a model pipeline can fit, predict, be saved, and re-loaded."""
        # Create small synthetic dataset matching ALL_FEATURES
        n_samples = 10
        data = {}
        for col in ALL_FEATURES:
            if "lat" in col or "lon" in col or "km" in col or "deg" in col or "temp" in col or "wind" in col:
                data[col] = np.random.uniform(10.0, 50.0, n_samples)
            elif "sin" in col or "cos" in col:
                data[col] = np.random.uniform(-1.0, 1.0, n_samples)
            else:
                data[col] = np.random.randint(0, 2, n_samples)
                
        X = pd.DataFrame(data)
        y = pd.Series(np.random.uniform(100.0, 1000.0, n_samples), name=TARGET)

        # Build pipeline
        preprocessor = create_preprocessor()
        estimator = Ridge(alpha=1.0)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", estimator)
        ])

        # Fit
        pipeline.fit(X, y)
        preds_before = pipeline.predict(X)
        self.assertEqual(len(preds_before), n_samples)

        # Save to temp path
        temp_dir = Path("models/test_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        model_path = temp_dir / "test_model.joblib"
        
        try:
            joblib.dump(pipeline, model_path)
            self.assertTrue(model_path.exists())

            # Load
            loaded_pipeline = joblib.load(model_path)
            preds_after = loaded_pipeline.predict(X)

            # Assert identical predictions
            np.testing.assert_array_almost_equal(preds_before, preds_after)
        finally:
            # Clean up
            if model_path.exists():
                model_path.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
