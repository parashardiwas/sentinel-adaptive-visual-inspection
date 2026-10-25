"""
Telemetry Loader for Ingesting Step 1 (Vision) and Step 2 (RL Scheduler) Outputs
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
from rag.config import RAGConfig

class TelemetryLoader:
    def __init__(self, vision_dir: Path = None, scheduler_dir: Path = None):
        self.vision_dir = vision_dir or RAGConfig.get_vision_outputs_dir()
        self.scheduler_dir = scheduler_dir or RAGConfig.get_scheduler_outputs_dir()

    def load_inference_telemetry(self) -> pd.DataFrame:
        """
        Loads the combined operational inference dataset.
        Merges Vision anomaly scores with RL scheduler decisions.
        """
        sched_file = self.scheduler_dir / "scheduler_inference_ready.csv"
        vision_file = self.vision_dir / "current_anomaly_score.csv"

        if sched_file.exists():
            sched_df = pd.read_csv(sched_file)
            if vision_file.exists():
                vision_df = pd.read_csv(vision_file)
                required_vision_columns = {"method", "image_path", "anomaly_score", "normalized_score", "predicted_defect_type"}
                if not required_vision_columns.issubset(vision_df.columns) or "image_path" not in sched_df.columns:
                    sched_df.attrs["source"] = "scheduler_export_only"
                    return sched_df
                # Keep mahalanobis method rows from vision
                vision_mah = vision_df[vision_df["method"].astype(str).str.lower() == "mahalanobis"].copy()
                merged = pd.merge(
                    sched_df,
                    vision_mah[["image_path", "anomaly_score", "normalized_score", "predicted_defect_type"]],
                    on="image_path",
                    how="left"
                )
                merged.attrs["source"] = "scheduler_and_vision_exports"
                return merged
            # Scheduler exports are not detector output. Do not synthesize
            # anomaly scores or labels from scheduler risk tiers.
            sched_df.attrs["source"] = "scheduler_export_only"
            return sched_df

        empty = pd.DataFrame(columns=["image_path", "category", "risk_level", "selected_action"])
        empty.attrs["source"] = "unavailable"
        return empty

    def load_adaptation_telemetry(self) -> Dict[str, Any]:
        """Loads the RL adaptation experiment results and summaries."""
        results_file = self.scheduler_dir / "adaptation_demo_results.csv"
        summary_file = self.scheduler_dir / "adaptation_demo_summary.csv"

        if results_file.exists() and summary_file.exists():
            return {
                "results_df": pd.read_csv(results_file),
                "summary_df": pd.read_csv(summary_file),
                "source": "live_file"
            }

        return {
            "summary_df": pd.DataFrame(),
            "results_df": pd.DataFrame(),
            "source": "unavailable",
        }
