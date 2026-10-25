"""
Operator Copilot - Interactive RAG Assistant for Quality Technicians & Line Supervisors
Powered by Groq LLM with Two-Tier Caching & Deterministic Fallback.
"""

from typing import Dict, Any, Optional
import pandas as pd
from rag.retriever import RAGRetriever
from rag.telemetry_loader import TelemetryLoader
from rag.llm import LLMClient
from rag.cache import RAGCache

class OperatorCopilot:
    def __init__(
        self,
        retriever: RAGRetriever,
        telemetry_loader: TelemetryLoader = None,
        llm_client: LLMClient = None,
        cache: RAGCache = None
    ):
        self.retriever = retriever
        self.loader = telemetry_loader or TelemetryLoader()
        self.llm = llm_client or LLMClient()
        self.cache = cache or RAGCache()
        self._telemetry_df: Optional[pd.DataFrame] = None

    @property
    def telemetry_df(self) -> pd.DataFrame:
        if self._telemetry_df is None:
            self._telemetry_df = self.loader.load_inference_telemetry()
        return self._telemetry_df

    def diagnose_image(self, image_path: str) -> Dict[str, Any]:
        """
        Retrieves telemetry for a specific image, queries the SOP knowledge base,
        and generates an end-to-end diagnostic explanation.
        """
        if self.telemetry_df.empty or "image_path" not in self.telemetry_df.columns:
            return {
                "status": "unavailable",
                "message": "No inspection telemetry is available to locate this image. SOP questions can still be answered.",
            }

        matches = self.telemetry_df[self.telemetry_df["image_path"].astype(str).str.contains(image_path, regex=False)]
        if matches.empty:
            filename = image_path.split("/")[-1]
            matches = self.telemetry_df[self.telemetry_df["image_path"].str.endswith(filename)]

        if matches.empty:
            return {
                "status": "error",
                "message": f"Image path '{image_path}' not found in telemetry records."
            }

        row = matches.iloc[0].to_dict()
        category = str(row.get("category", "general")).lower()
        defect_type = str(row.get("predicted_defect_type", "general")).lower()
        risk_level = str(row.get("risk_level", "medium")).lower()
        selected_action = str(row.get("selected_action", "prediction")).lower()
        norm_score = float(row.get("normalized_score", 0.0))

        # Check cache
        cache_key = f"diag::{image_path}::{defect_type}::{selected_action}"
        cached = self.cache.get(cache_key, category)
        if cached:
            return cached

        # Retrieve relevant SOP documentation
        query = f"rework root cause severity disposal for {defect_type} in {category}"
        retrieved_chunks = self.retriever.retrieve(
            query=query,
            category=category,
            defect_code=defect_type if defect_type != "good" else None,
            top_k=2
        )
        formatted_sop = self.retriever.format_context_for_prompt(retrieved_chunks)

        if self.llm.is_available():
            sys_prompt = (
                "You are an expert Manufacturing Quality Assurance Engineer on the factory floor. "
                "Synthesize the visual inspection score, RL scheduler decision, and the retrieved official SOP "
                "into a concise, authoritative diagnosis. Cite the specific SOP document ID. "
                "State: 1) What was detected, 2) Why RL took this action, 3) Immediate rework vs scrap disposition, "
                "and 4) Machine preventive maintenance action."
            )
            user_prompt = (
                f"Part ID: {row.get('image_path')}\n"
                f"Category: {category.upper()}\n"
                f"Vision Normalized Score: {norm_score:.3f} (Threshold ~ 0.95)\n"
                f"Predicted Defect: {defect_type}\n"
                f"RL Policy Action: {selected_action.upper()} ({row.get('recommended_next_step')})\n\n"
                f"Official SOP Documentation:\n{formatted_sop}\n\n"
                f"Provide the diagnostic briefing:"
            )
            explanation = self.llm.generate(prompt=user_prompt, system_prompt=sys_prompt)
        else:
            explanation = self._build_operator_explanation(
                row=row,
                category=category,
                defect_type=defect_type,
                risk_level=risk_level,
                selected_action=selected_action,
                norm_score=norm_score,
                formatted_sop=formatted_sop
            )

        res = {
            "status": "success",
            "image_path": row.get("image_path"),
            "category": category,
            "predicted_defect_type": defect_type,
            "normalized_score": norm_score,
            "risk_level": risk_level,
            "selected_action": selected_action,
            "recommended_next_step": row.get("recommended_next_step"),
            "retrieved_sop": formatted_sop,
            "operator_guidance": explanation,
            "llm_backend": self.llm.backend
        }
        self.cache.set(cache_key, res, category)
        return res

    def ask(self, question: str, category: Optional[str] = None) -> Dict[str, Any]:
        """
        General operator inquiry answered with grounded citations from the manufacturing SOPs.
        """
        cached = self.cache.get(question, category)
        if cached:
            return cached

        chunks = self.retriever.retrieve(
            query=question,
            category=category,
            top_k=3
        )
        context = self.retriever.format_context_for_prompt(chunks)

        if self.llm.is_available():
            sys_prompt = (
                "You are an expert Quality Control & Process Engineer on an adaptive inspection line. "
                "Answer the operator's question using ONLY the provided Standard Operating Procedure (SOP) documentation. "
                "Always cite the exact Document ID (e.g. SOP-GRD-02, SOP-ZIP-03) and section. "
                "Be direct, actionable, and focus on shop-floor quality compliance."
            )
            user_prompt = (
                f"Operator Question: {question}\n"
                f"Target Category: {category or 'All Categories'}\n\n"
                f"Retrieved SOP Context:\n{context}\n\n"
                f"Provide the engineering response:"
            )
            answer = self.llm.generate(prompt=user_prompt, system_prompt=sys_prompt)
        else:
            answer = (
                f"### Grounded Engineering Response\n\n"
                f"Based on the official Standard Operating Procedures retrieved for your query:\n\n"
                f"{context}\n\n"
                f"**Operational Note**: All procedural actions must be logged in the Shift Quality Ledger."
            )

        res = {
            "question": question,
            "category": category,
            "retrieved_chunks_count": len(chunks),
            "response": answer,
            "llm_backend": self.llm.backend
        }
        self.cache.set(question, res, category)
        return res

    def _build_operator_explanation(
        self,
        row: dict,
        category: str,
        defect_type: str,
        risk_level: str,
        selected_action: str,
        norm_score: float,
        formatted_sop: str
    ) -> str:
        return f"""
### Diagnostic Inspection Summary
- **Part Identifier**: `{row.get('image_path')}`
- **Product Category**: `{category.upper()}`
- **Vision Evaluation**: Normalized Anomaly Score = `{norm_score:.3f}` (Threshold ~ 0.95)
- **Identified Defect Type**: `{defect_type}`
- **RL Risk Tier**: `{risk_level.upper()}`
- **RL Action Executed**: `{selected_action.upper()}` ({row.get('recommended_next_step')})

### Actionable Remediation Guidelines (Ref: Official Manufacturing SOP)
{formatted_sop}
"""
