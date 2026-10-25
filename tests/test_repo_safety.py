import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rag.embeddings import EmbeddingModel
from rag.report_generator import QualityReportGenerator
from rag.telemetry_loader import TelemetryLoader
from rag.vector_store import VectorStore
from rag.config import RAGConfig
from unittest.mock import patch


class TelemetryLoaderTests(unittest.TestCase):
    def test_missing_exports_are_empty_and_marked_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            loader = TelemetryLoader(root / "vision", root / "scheduler")

            telemetry = loader.load_inference_telemetry()
            adaptation = loader.load_adaptation_telemetry()

        self.assertTrue(telemetry.empty)
        self.assertEqual(telemetry.attrs["source"], "unavailable")
        self.assertEqual(adaptation["source"], "unavailable")
        self.assertTrue(adaptation["summary_df"].empty)

    def test_scheduler_export_does_not_create_detector_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scheduler = root / "scheduler"
            scheduler.mkdir()
            pd.DataFrame({
                "image_path": ["grid/test/bent/001.png"],
                "category": ["grid"],
                "risk_level": ["high"],
                "selected_action": ["expert"],
            }).to_csv(scheduler / "scheduler_inference_ready.csv", index=False)
            telemetry = TelemetryLoader(root / "vision", scheduler).load_inference_telemetry()

        self.assertEqual(telemetry.attrs["source"], "scheduler_export_only")
        self.assertNotIn("anomaly_score", telemetry.columns)
        self.assertNotIn("is_anomaly", telemetry.columns)


class RAGCacheTests(unittest.TestCase):
    def test_fallback_features_are_repeatable(self):
        encoder = EmbeddingModel.__new__(EmbeddingModel)
        vectors = encoder._fallback_encode(["bent wire fracture", "bent wire fracture"])
        np.testing.assert_array_equal(vectors[0], vectors[1])
        self.assertEqual(EmbeddingModel._stable_bucket("fracture", 384), EmbeddingModel._stable_bucket("fracture", 384))

    def test_vector_store_rejects_stale_encoder_or_corpus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = VectorStore(Path(temp_dir))
            chunks = [{"id": "one", "content": "wire fracture", "category": "grid"}]
            store.add_documents(chunks, np.ones((1, 384), dtype=np.float32))
            store.persist(fingerprint="encoder-a:corpus-a")

            self.assertTrue(VectorStore(Path(temp_dir)).load(expected_fingerprint="encoder-a:corpus-a"))
            self.assertFalse(VectorStore(Path(temp_dir)).load(expected_fingerprint="encoder-b:corpus-a"))
            self.assertFalse(VectorStore(Path(temp_dir)).load(expected_fingerprint="encoder-a:corpus-b"))

    def test_empty_report_does_not_claim_zero_cost_or_savings(self):
        class EmptyLoader:
            def load_inference_telemetry(self):
                return pd.DataFrame()

            def load_adaptation_telemetry(self):
                return {"summary_df": pd.DataFrame(), "results_df": pd.DataFrame(), "source": "unavailable"}

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with patch.object(RAGConfig, "get_rag_outputs_dir", return_value=output_dir):
                report = QualityReportGenerator(retriever=None, telemetry_loader=EmptyLoader()).generate_shift_report()

        self.assertIsNone(report["total_items"])
        self.assertIsNone(report["cost_savings_pct"])
        self.assertIn("Operational telemetry unavailable", report["markdown_report"])
        self.assertNotIn("0.0% savings", report["markdown_report"])


if __name__ == "__main__":
    unittest.main()
