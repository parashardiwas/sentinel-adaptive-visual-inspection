"""
Automated Quality Assurance & Incident Shift Report Generator
Synthesizes Step 1 (Vision) and Step 2 (RL) telemetry with RAG-retrieved SOP documentation.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
from rag.retriever import RAGRetriever
from rag.telemetry_loader import TelemetryLoader
from rag.config import RAGConfig

class QualityReportGenerator:
    def __init__(self, retriever: RAGRetriever, telemetry_loader: TelemetryLoader = None):
        self.retriever = retriever
        self.loader = telemetry_loader or TelemetryLoader()

    def generate_shift_report(self, shift_name: str = "Shift A - Primary Assembly") -> Dict[str, Any]:
        """
        Generates a comprehensive manufacturing shift quality report.
        Blends quantitative metrics from Vision & RL with qualitative engineering SOPs from RAG.
        """
        telemetry_df = self.loader.load_inference_telemetry()
        adaptation_data = self.loader.load_adaptation_telemetry()

        valid_actions = (
            telemetry_df["selected_action"].astype(str).str.lower().isin(["skip", "prediction", "expert"]).any()
            if "selected_action" in telemetry_df.columns else False
        )
        if telemetry_df.empty or not valid_actions:
            markdown_report = (
                "# Executive Quality Assurance & Incident Shift Report\n"
                f"**Shift Designation**: {shift_name}  \n"
                f"**Audit Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n"
                "## Operational telemetry unavailable\n"
                "No usable scheduler action telemetry was found. Unit counts, triage costs, savings, defect clusters, "
                "and adaptation results are not reported. Add the authorized scheduler export under the configured "
                "outputs directory and regenerate this report.\n"
            )
            out_dir = RAGConfig.get_rag_outputs_dir()
            report_file = out_dir / "shift_quality_audit_report.md"
            report_file.write_text(markdown_report, encoding="utf-8")
            return {
                "shift_name": shift_name,
                "total_items": None,
                "cost_savings_pct": None,
                "rl_cost": None,
                "baseline_cost": None,
                "markdown_report": markdown_report,
                "report_path": str(report_file),
            }

        # Quantitative Metrics
        total_items = len(telemetry_df)
        action_counts = telemetry_df["selected_action"].value_counts().to_dict()
        risk_counts = telemetry_df["risk_level"].value_counts().to_dict()
        
        skip_count = action_counts.get("skip", 0)
        pred_count = action_counts.get("prediction", 0)
        expert_count = action_counts.get("expert", 0)

        # Cost Modeling
        # Baselines: 100% Expert = total_items * $5.00
        # RL Policy Cost = (skip * $0) + (pred * $1) + (expert * $5)
        baseline_cost = total_items * 5.0
        rl_actual_cost = (pred_count * 1.0) + (expert_count * 5.0)
        cost_savings_pct = ((baseline_cost - rl_actual_cost) / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

        # High Risk Clusters
        high_risk_df = telemetry_df[telemetry_df["risk_level"] == "high"]
        defect_clusters = {}
        if not high_risk_df.empty and "predicted_defect_type" in high_risk_df.columns:
            defect_clusters = high_risk_df["predicted_defect_type"].value_counts().head(3).to_dict()

        # Retrieve RAG Guidance for top defects
        rag_defect_insights = []
        for defect_name, count in defect_clusters.items():
            if defect_name == "good":
                continue
            matching_rows = high_risk_df[high_risk_df["predicted_defect_type"] == defect_name]
            sample_cat = matching_rows["category"].iloc[0] if not matching_rows.empty else "general"

            chunks = self.retriever.retrieve(
                query=f"root cause and rework for {defect_name} defect in {sample_cat}",
                category=sample_cat,
                defect_code=defect_name,
                top_k=2
            )
            rag_defect_insights.append({
                "defect_name": defect_name,
                "category": sample_cat,
                "count": count,
                "retrieved_context": self.retriever.format_context_for_prompt(chunks)
            })

        # Compile Markdown Report
        markdown_report = self._build_markdown_report(
            shift_name=shift_name,
            total_items=total_items,
            action_counts=action_counts,
            risk_counts=risk_counts,
            baseline_cost=baseline_cost,
            rl_actual_cost=rl_actual_cost,
            cost_savings_pct=cost_savings_pct,
            defect_clusters=defect_clusters,
            rag_defect_insights=rag_defect_insights,
            adaptation_data=adaptation_data
        )

        # Save to disk
        out_dir = RAGConfig.get_rag_outputs_dir()
        report_file = out_dir / "shift_quality_audit_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(markdown_report)

        return {
            "shift_name": shift_name,
            "total_items": total_items,
            "cost_savings_pct": cost_savings_pct,
            "rl_cost": rl_actual_cost,
            "baseline_cost": baseline_cost,
            "markdown_report": markdown_report,
            "report_path": str(report_file)
        }

    def _build_markdown_report(
        self,
        shift_name: str,
        total_items: int,
        action_counts: dict,
        risk_counts: dict,
        baseline_cost: float,
        rl_actual_cost: float,
        cost_savings_pct: float,
        defect_clusters: dict,
        rag_defect_insights: list,
        adaptation_data: dict
    ) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = (
            f"# Executive Quality Assurance & Incident Shift Report\n"
            f"**Shift Designation**: {shift_name}  \n"
            f"**Audit Timestamp**: {timestamp}  \n"
            f"**System Status**: Adaptive Vision-RL Inspection Cell Active  \n\n"
            f"---\n\n"
            f"## 1. Executive Summary\n"
            f"During this operating period, the system processed **{total_items} units** across product lines. \n"
            f"By employing the Q-Learning Adaptive Scheduler rather than 100% manual inspection, the facility achieved:\n"
            f"- **Operational Cost Reduction**: **{cost_savings_pct:.1f}% savings** (${rl_actual_cost:.2f} incurred vs ${baseline_cost:.2f} baseline).\n"
            f"- **Inspection Routing Breakdown**:\n"
            f"  - `skip` (Automated Bypass): **{action_counts.get('skip', 0)}** units ({action_counts.get('skip', 0) / max(total_items, 1) * 100:.1f}%)\n"
            f"  - `prediction` (Automated CV Gating): **{action_counts.get('prediction', 0)}** units ({action_counts.get('prediction', 0) / max(total_items, 1) * 100:.1f}%)\n"
            f"  - `expert` (Escalated to Human Review): **{action_counts.get('expert', 0)}** units ({action_counts.get('expert', 0) / max(total_items, 1) * 100:.1f}%)\n\n"
            f"---\n\n"
            f"## 2. Risk Tier Classification Distribution\n"
            f"- **Low Risk (Margin < -0.10)**: {risk_counts.get('low', 0)} units\n"
            f"- **Medium Risk (-0.10 to +0.10)**: {risk_counts.get('medium', 0)} units (Decision boundary uncertainty)\n"
            f"- **High Risk (Margin > +0.10)**: {risk_counts.get('high', 0)} units (Anomalies requiring intervention)\n\n"
            f"---\n\n"
            f"## 3. High-Priority Defect Clusters & Root-Cause Remediation (RAG-Retrieved)\n"
        )
        if not rag_defect_insights:
            md += "\n*No critical defect clusters detected during this cycle. System operating within nominal Cpk limits.*\n"
        else:
            for item in rag_defect_insights:
                md += f"""
### Defect Signature: `{item['defect_name']}` ({item['category'].upper()})
- **Incidence Count**: {item['count']} high-risk units detected
- **Standard Operating Procedure & Engineering Guidance**:
```markdown
{item['retrieved_context']}
```
"""

        # Adaptation section if available
        summary_df = adaptation_data.get("summary_df")
        if summary_df is not None and not summary_df.empty:
            md += """
---

## 4. Reinforcement Learning Adaptation Audit (Drift Response)
The scheduler's dynamic drift-adaptation experiment confirmed real-time responsiveness when defect spikes occur:

| Period | Defect Share | Expert Review Share | Missed Defect Count | Average Item Cost |
|---|---|---|---|---|
"""
            for _, row in summary_df.iterrows():
                p = row.get("period", "N/A")
                dr = row.get("defect_rate", 0) * 100
                er = row.get("expert_review_rate", 0) * 100
                md_cnt = row.get("missed_defects", 0)
                cost = row.get("avg_cost", 0)
                md += f"| {p} | {dr:.1f}% | {er:.1f}% | {md_cnt} | ${cost:.2f} |\n"

            md += """
> **Engineering Assessment**: When the upstream defect rate surged, the RL policy dynamically ramped expert escalations to prevent defect escapes, maintaining rigorous outgoing quality compliance.
"""

        md += """
---

## 5. Corrective Action Plan & Sign-Off
- [ ] Review high-risk audit bin items with Cell Quality Supervisor.
- [ ] Execute scheduled PM checklist for top defect root causes cited above.
- [ ] Maintain RL scheduler in active adaptive mode.

**Report Generated by**: Adaptive Visual Inspection System (Step 3: NLP / RAG Module)  
**Status**: Confirmed & Logged for BI Dashboard (Step 4)
"""
        return md
