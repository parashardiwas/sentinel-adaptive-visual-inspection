"""
Dashboard Adapter for RAG Engine & Model Benchmarking
Seamlessly connects Streamlit Studio (app.py) with the RAG Knowledge Base and Copilot.
"""

import streamlit as st
import pandas as pd
import hashlib
import json
from pathlib import Path
from typing import Dict, Any, List

from rag.ingestion import KnowledgeBaseIngestor
from rag.embeddings import EmbeddingModel
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever
from rag.telemetry_loader import TelemetryLoader
from rag.report_generator import QualityReportGenerator
from rag.copilot import OperatorCopilot
from rag.llm import LLMClient
from rag.cache import RAGCache
from inference2 import BUILTIN_CATEGORY_THRESHOLDS

@st.cache_resource(show_spinner="Initializing Manufacturing SOP Knowledge Base & Vector Index...")
def get_rag_components():
    """Initializes and caches the RAG engine once for the Streamlit session."""
    ingestor = KnowledgeBaseIngestor()
    chunks = ingestor.load_documents()

    embed_model = EmbeddingModel()
    vector_store = VectorStore()

    corpus_identity = json.dumps(chunks, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    cache_fingerprint = hashlib.sha256(
        f"{embed_model.cache_fingerprint}:{corpus_identity}".encode("utf-8")
    ).hexdigest()

    if not vector_store.load(expected_fingerprint=cache_fingerprint):
        vector_store = VectorStore()
        embeddings = embed_model.encode([c["content"] for c in chunks])
        vector_store.add_documents(chunks, embeddings)
        vector_store.persist(fingerprint=cache_fingerprint)

    retriever = RAGRetriever(vector_store, embed_model)
    llm_client = LLMClient()
    cache = RAGCache()

    copilot = OperatorCopilot(retriever, llm_client=llm_client, cache=cache)
    report_gen = QualityReportGenerator(retriever)

    return {
        "retriever": retriever,
        "copilot": copilot,
        "report_gen": report_gen,
        "llm_client": llm_client,
        "chunks_count": len(chunks)
    }

def render_rag_audit_tab(result: Dict[str, Any], category: str, key_prefix: str = "quality-audit"):
    """Renders the executive quality assurance & incident shift audit report."""
    st.markdown("### Executive Quality Assurance & Incident Shift Report")
    st.caption("Synthesizes real-time frame telemetry with official MVTec industrial Standard Operating Procedures (SOPs).")

    rag = get_rag_components()
    table = result.get("table", pd.DataFrame())

    is_batch_data = False
    if table.empty:
        # Graceful fallback: load batch telemetry from Step 1 / Step 2
        batch_df = TelemetryLoader().load_inference_telemetry()
        if not batch_df.empty:
            cat_df = batch_df[batch_df["category"].astype(str).str.lower() == category.lower()]
            table = cat_df if not cat_df.empty else batch_df
            is_batch_data = True
        else:
            st.info("No live inspection frames or exported scheduler telemetry are available. Operational metrics are unavailable; SOP guidance and the copilot remain available.")
            return

    if is_batch_data:
        st.info(f"Currently auditing **MVTec `{category.upper()}` shift batch data**. Upload a video in Inspection to audit sampled frames from that session.")

    total_frames = len(table)
    score_name = "normalized_score" if "normalized_score" in table.columns else "anomaly_score"
    score_col = pd.to_numeric(table[score_name], errors="coerce") if score_name in table.columns else pd.Series(float("nan"), index=table.index)
    summary = result.get("report", {}) or {}
    fixed_boundary = summary.get("decision_boundary", result.get("anomaly_score_threshold_for_summary"))
    if fixed_boundary is not None:
        anomaly_boundaries = pd.Series(float(fixed_boundary), index=table.index)
    elif "category" in table.columns:
        anomaly_boundaries = table["category"].astype(str).str.lower().map(BUILTIN_CATEGORY_THRESHOLDS).fillna(0.95)
    else:
        anomaly_boundaries = pd.Series(float(BUILTIN_CATEGORY_THRESHOLDS.get(category.lower(), 0.95)), index=table.index)
    has_scores = bool(score_col.notna().any())
    anom_count = int((score_col >= anomaly_boundaries).sum()) if has_scores else None
    
    # Action counts
    if "ppo_final_action" in table.columns:
        actions = table["ppo_final_action"].astype(str).value_counts().to_dict()
    elif "selected_action" in table.columns:
        action_map = {"skip": "MONITOR", "prediction": "FLAG_ANOMALY", "expert": "HUMAN_REVIEW"}
        actions = table["selected_action"].astype(str).str.lower().map(action_map).value_counts().to_dict()
    else:
        actions = {}
    has_actions = any(action in actions for action in ("MONITOR", "FLAG_ANOMALY", "HUMAN_REVIEW"))
    monitor_count = actions.get("MONITOR", 0)
    flag_count = actions.get("FLAG_ANOMALY", 0)
    review_count = actions.get("HUMAN_REVIEW", 0)

    # Cost calculation: 100% human review baseline ($5/frame) vs RL policy
    baseline_cost = total_frames * 5.0 if has_actions else None
    rl_cost = (flag_count * 1.0) + (review_count * 5.0) if has_actions else None
    savings_pct = ((baseline_cost - rl_cost) / max(baseline_cost, 1)) * 100.0 if has_actions else None

    # Primary detected defect subtype
    top_subtype = "general"
    if "predicted_subtype" in table.columns:
        subtypes = table["predicted_subtype"].dropna().value_counts()
        if not subtypes.empty:
            top_subtype = subtypes.index[0]

    # Retrieve relevant SOP
    chunks = rag["retriever"].retrieve(
        query=f"root cause rework disposal preventive maintenance for {top_subtype} in {category}",
        category=category,
        defect_code=top_subtype if top_subtype != "general" else None,
        top_k=2
    )
    sop_context = rag["retriever"].format_context_for_prompt(chunks)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sampled Frames", f"{total_frames:,}")
    m2.metric("Anomalous Frames", f"{anom_count:,}" if anom_count is not None else "Unavailable", delta=f"{anom_count/max(total_frames,1)*100:.1f}% rate" if anom_count is not None else None, delta_color="inverse")
    m3.metric("Cost Savings vs 100% Audit", f"{savings_pct:.1f}%" if savings_pct is not None else "Unavailable", delta=f"${baseline_cost - rl_cost:.2f} saved" if savings_pct is not None else None)
    m4.metric("LLM Reasoner", rag["llm_client"].backend.upper())

    st.markdown("---")

    # Executive Narrative
    if is_batch_data:
        source_label = "Held-out Step 1 / Step 2 batch telemetry"
        policy_label = "Recorded scheduler actions from the held-out batch"
    else:
        source_label = "Active inspection session"
        policy_mode = str(summary.get("ppo_policy_mode", ""))
        if policy_mode == "v4_recall_balanced":
            policy_label = "Learned PPO V4 policy"
        elif policy_mode == "legacy_transformer":
            policy_label = "Legacy PPO policy"
        elif policy_mode == "adaptive_triage_fallback":
            policy_label = "Adaptive triage fallback (no PPO checkpoint)"
        elif summary.get("inspection_mode", "").startswith("Native MVTec Image Inspection"):
            policy_label = "Score-based single-image triage (not PPO/Q-Learning)"
        else:
            policy_label = "Visual scoring only; no RL action policy reported"

    if has_actions:
        action_summary = (
            f"- **Automated Pass (`MONITOR` / `skip`)**: {monitor_count} frames ({monitor_count/max(total_frames,1)*100:.1f}%)\n"
            f"- **Automated Model Flag (`FLAG_ANOMALY` / `prediction`)**: {flag_count} frames ({flag_count/max(total_frames,1)*100:.1f}%)\n"
            f"- **Human Expert Review (`HUMAN_REVIEW` / `expert`)**: {review_count} frames ({review_count/max(total_frames,1)*100:.1f}%)"
        )
        cost_summary = f"Total incurred inspection cost **${rl_cost:.2f}** compared to **${baseline_cost:.2f}** for manual audit (**{savings_pct:.1f}% savings**)."
    else:
        action_summary = "Recorded triage actions are unavailable in this data source."
        cost_summary = "Unavailable because recorded scheduler actions were not included."

    report_md = f"""## Inspection Incident & Quality Summary
**Product Category**: `{category.upper()}`  
**Data Source**: {source_label}

**Triage Policy**: {policy_label}

### 1. Throughput & Triage Breakdown
{action_summary}
- **Cost Performance**: {cost_summary}

### 2. High-Priority Defect Diagnosis & Official SOP Guidance
- **Primary Identified Defect Signature**: `{top_subtype}`
{sop_context}

### 3. Action Items & Quality Disposition
- [ ] Pull suspect batch units flagged for `HUMAN_REVIEW` to secondary containment rack.
- [ ] Review tooling wear according to preventive maintenance guidance cited above.
- [ ] Log shift metrics in ERP quality module.
"""
    st.markdown(report_md)

    st.download_button(
        "Download Executive Shift Audit Report (.md)",
        data=report_md.encode("utf-8"),
        file_name=f"sentinel_{category}_shift_quality_audit.md",
        mime="text/markdown",
        width="stretch",
        key=f"{key_prefix}-report-download",
    )

def render_copilot_tab(result: Dict[str, Any], category: str, key_prefix: str = "copilot"):
    """Renders the interactive quality technician assistant."""
    st.markdown("### Interactive Quality Control Operator Copilot")
    st.caption("Ask questions about defect causes, rework instructions, machine maintenance, or specific inspected frames.")

    rag = get_rag_components()
    table = result.get("table", pd.DataFrame())

    # Section 1: Frame-Specific Inspection Assistant
    if not table.empty:
        if "ppo_final_action" in table.columns:
            review_frames = table[table["ppo_final_action"].astype(str).isin(["FLAG_ANOMALY", "HUMAN_REVIEW"])]
        else:
            review_frames = pd.DataFrame()
        if review_frames.empty:
            review_frames = table.sort_values(by="anomaly_score", ascending=False).head(5)

        st.markdown("#### Escalate / Inspect Specific Frame")
        frame_options = review_frames["frame_index"].tolist()
        if frame_options:
            selected_frame = st.selectbox(
                "Select flagged frame to inspect:",
                frame_options,
                format_func=lambda f: f"Frame {f} (Score: {table.loc[table['frame_index']==f, 'anomaly_score'].iloc[0]:.3f} | Action: {table.loc[table['frame_index']==f, 'ppo_final_action'].iloc[0] if 'ppo_final_action' in table.columns else 'N/A'})",
                key=f"{key_prefix}-frame-select",
            )
            
            row = table[table["frame_index"] == selected_frame].iloc[0]
            score = float(row.get("anomaly_score", 0.0))
            action = str(row.get("ppo_final_action", "FLAG_ANOMALY"))
            subtype = str(row.get("predicted_subtype", "defect"))

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**Frame {selected_frame} Telemetry**")
                st.write(f"- Category: `{category}`")
                st.write(f"- Score: `{score:.4f}`")
                st.write(f"- Subtype: `{subtype}`")
                st.write(f"- Action: `{action}`")
            with col2:
                # Instant diagnosis
                chunks = rag["retriever"].retrieve(
                    query=f"rework procedure root cause for {subtype} in {category}",
                    category=category,
                    defect_code=subtype if subtype != "defect" else None,
                    top_k=2
                )
                st.info(f"**Official SOP Directive ({chunks[0]['doc_id'] if chunks else 'SOP-GEN-01'}):**\n\n" + (chunks[0]["content"][:350] + "..." if chunks else "Follow standard inspection protocol."))

    st.markdown("---")
    st.markdown("#### Ask Quality Engineering Copilot")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {"role": "assistant", "content": f"Hello! I am your Quality Control Copilot grounded in the official `{category.upper()}` standard operating procedures. How can I assist you with this batch?"}
        ]

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input(
        "Ask about scrap criteria, root causes, tooling adjustments...",
        key=f"{key_prefix}-chat-input",
    )
    if user_query:
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Consulting Manufacturing SOP Knowledge Base..."):
                resp = rag["copilot"].ask(user_query, category=category)
                reply = resp.get("response", "No relevant guidance found.")
                st.markdown(reply)
                st.session_state["chat_history"].append({"role": "assistant", "content": reply})

def render_benchmark_tab(result: Dict[str, Any], category: str, key_prefix: str = "benchmark"):
    """Renders side-by-side comparison between Pipeline A (Q-Learning) and Pipeline B (PPO V4)."""
    import altair as alt

    st.markdown("### Side-by-Side Model Architecture Benchmark")
    st.caption("Displays recorded action telemetry only. A live dual-pipeline comparison requires independent Pipeline A and Pipeline B outputs on the same records.")

    session_table = result.get("table", pd.DataFrame())
    # A live comparison requires outputs from both schedulers on the same records.
    # A Pipeline B-only video or single-image result is not a dual-pipeline benchmark.
    has_session = (
        not session_table.empty
        and {"selected_action", "ppo_final_action"}.issubset(session_table.columns)
    )

    bench_options = ["Held-Out Validation Benchmark (101 Records)", "Active Session Frames"] if has_session else ["Held-Out Validation Benchmark (101 Records)"]
    data_scope = st.radio(
        "Benchmark Scope:",
        bench_options,
        horizontal=True,
        key=f"{key_prefix}-data-scope-radio",
    )

    if "Held-Out" in data_scope:
        raw_df = TelemetryLoader().load_inference_telemetry()
        if raw_df.empty or "category" not in raw_df.columns:
            st.info("No held-out telemetry export is available. Run an inspection session or provide the scheduler export to enable this benchmark.")
            return
        all_label = f"All Categories ({len(raw_df)} records)"
        categories = [all_label] + [c.title() for c in sorted(raw_df["category"].unique())]
        selected_cat = st.selectbox(
            "Filter Benchmark Category:", categories, index=0,
            key=f"{key_prefix}-category-filter",
        )
        if selected_cat != all_label:
            table = raw_df[raw_df["category"].str.lower() == selected_cat.lower()].copy()
        else:
            table = raw_df.copy()
        st.info(f"Displaying **{len(table)} held-out scheduler records** from the offline export. These are recorded telemetry, not current model inference.")
    else:
        table = session_table.copy()
        st.info(f"Displaying **{len(table)} live inspection frames** from current inspection run.")

    total = len(table)
    if total == 0:
        st.warning("No records available for the selected benchmark filter.")
        return

    has_pipeline_a_actions = (
        "selected_action" in table.columns
        and table["selected_action"].astype(str).str.lower().isin(["skip", "prediction", "expert"]).any()
    )
    has_pipeline_b_actions = (
        "ppo_final_action" in table.columns
        and table["ppo_final_action"].astype(str).str.upper().isin(["MONITOR", "FLAG_ANOMALY", "HUMAN_REVIEW"]).any()
    )

    # Use only actions actually present in the held-out/live records. Do not infer
    # a Pipeline B policy from Pipeline A's Q-learning decisions or from score tiers.
    if has_pipeline_b_actions:
        ppo_monitor = int((table["ppo_final_action"] == "MONITOR").sum())
        ppo_flag = int((table["ppo_final_action"] == "FLAG_ANOMALY").sum())
        ppo_review = int((table["ppo_final_action"] == "HUMAN_REVIEW").sum())
    else:
        ppo_monitor = ppo_flag = ppo_review = 0

    if has_pipeline_a_actions:
        q_skip = int((table["selected_action"] == "skip").sum())
        q_pred = int((table["selected_action"] == "prediction").sum())
        q_expert = int((table["selected_action"] == "expert").sum())
    else:
        q_skip = q_pred = q_expert = 0

    # Cost calculations ($5 manual baseline)
    cost_baseline = total * 5.0
    cost_q = (q_pred * 1.0) + (q_expert * 5.0) if has_pipeline_a_actions else None
    cost_ppo = (ppo_flag * 1.0) + (ppo_review * 5.0) if has_pipeline_b_actions else None
    savings_q = ((cost_baseline - cost_q) / max(cost_baseline, 1)) * 100.0 if cost_q is not None else None
    savings_ppo = ((cost_baseline - cost_ppo) / max(cost_baseline, 1)) * 100.0 if cost_ppo is not None else None

    agreement_pct = None
    if {"selected_action", "ppo_final_action"}.issubset(table.columns):
        q_map = {"skip": "MONITOR", "prediction": "FLAG_ANOMALY", "expert": "HUMAN_REVIEW"}
        q_actions = table["selected_action"].astype(str).str.lower().map(q_map)
        ppo_actions = table["ppo_final_action"].astype(str).str.upper()
        valid = q_actions.notna() & ppo_actions.notna()
        if valid.any():
            agreement_pct = float((q_actions[valid].to_numpy() == ppo_actions[valid].to_numpy()).mean() * 100.0)

    # ── High Level KPIs ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Evaluated Units", f"{total:,}")
    k2.metric(
        "Pipeline A Incurred Cost", f"${cost_q:.2f}" if cost_q is not None else "Unavailable",
        delta=f"{savings_q:.1f}% saved vs audit" if savings_q is not None else None,
    )
    k3.metric(
        "Pipeline B Incurred Cost", f"${cost_ppo:.2f}" if cost_ppo is not None else "Unavailable",
        delta=f"{savings_ppo:.1f}% saved vs audit" if savings_ppo is not None else None,
    )
    k4.metric(
        "Policy Concurrence", f"{agreement_pct:.1f}%" if agreement_pct is not None else "Unavailable",
        help="Row-by-row agreement on records with both recorded scheduler actions.",
    )

    st.markdown("---")

    # ── Architecture Side-by-Side Breakdown ──
    b1, b2 = st.columns(2, gap="large")
    with b1:
        st.markdown("<div class='panel' style='border-left:3px solid #89b8ca'><b>Pipeline A · Statistical anomaly detection</b></div>", unsafe_allow_html=True)
        if has_pipeline_a_actions:
            st.write("- **Scheduler data**: recorded `selected_action` values from the Step 2 export; this benchmark does not rerun Pipeline A inference.")
            st.write(f"- `skip` rate: **{q_skip/max(total,1)*100:.1f}%** ({q_skip} units · $0 cost)")
            st.write(f"- `prediction` rate: **{q_pred/max(total,1)*100:.1f}%** ({q_pred} units · $1 cost)")
            st.write(f"- `expert` rate: **{q_expert/max(total,1)*100:.1f}%** ({q_expert} units · $5 cost)")
        else:
            st.info("Pipeline A actions are not included in this record set.")

    with b2:
        st.markdown("<div class='panel' style='border-left:3px solid #69c9d4'><b>Pipeline B · Deep visual inspection</b></div>", unsafe_allow_html=True)
        st.write("- **Perception**: STFPM Feature Pyramid (Heatmaps) / YOLO26 Global + PatchCore Local")
        st.write("- **Subtyping**: DINOv2 (`vits14`) Vision Transformer + LinearSVM")
        if has_pipeline_b_actions:
            st.write("- **Scheduler data**: recorded Pipeline B `ppo_final_action` values.")
            st.write(f"- `MONITOR` rate: **{ppo_monitor/max(total,1)*100:.1f}%** ({ppo_monitor} units · $0 cost)")
            st.write(f"- `FLAG_ANOMALY` rate: **{ppo_flag/max(total,1)*100:.1f}%** ({ppo_flag} units · $1 cost)")
            st.write(f"- `HUMAN_REVIEW` rate: **{ppo_review/max(total,1)*100:.1f}%** ({ppo_review} units · $5 cost)")
        else:
            st.info("Pipeline B action records are not present. No PPO action distribution, cost, or agreement is inferred from Pipeline A telemetry.")

    st.markdown("---")

    # ── Comparative Visual Chart ──
    st.markdown("#### Triage Action Distribution")
    chart_data = []
    if has_pipeline_a_actions:
        chart_data.extend([
            {"Pipeline": "Pipeline A (recorded Q actions)", "Tier": "01 · Pass / Skip ($0)", "Count": q_skip},
            {"Pipeline": "Pipeline A (recorded Q actions)", "Tier": "02 · Automated Flag ($1)", "Count": q_pred},
            {"Pipeline": "Pipeline A (recorded Q actions)", "Tier": "03 · Expert Review ($5)", "Count": q_expert},
        ])
    if has_pipeline_b_actions:
        chart_data.extend([
            {"Pipeline": "Pipeline B (recorded PPO actions)", "Tier": "01 · Pass / Skip ($0)", "Count": ppo_monitor},
            {"Pipeline": "Pipeline B (recorded PPO actions)", "Tier": "02 · Automated Flag ($1)", "Count": ppo_flag},
            {"Pipeline": "Pipeline B (recorded PPO actions)", "Tier": "03 · Expert Review ($5)", "Count": ppo_review},
        ])
    cdf = pd.DataFrame(chart_data)
    pipeline_domain = []
    pipeline_colors = []
    if has_pipeline_a_actions:
        pipeline_domain.append("Pipeline A (recorded Q actions)")
        pipeline_colors.append("#89b8ca")
    if has_pipeline_b_actions:
        pipeline_domain.append("Pipeline B (recorded PPO actions)")
        pipeline_colors.append("#69c9d4")
    action_chart = alt.Chart(cdf).mark_bar().encode(
        x=alt.X("Tier:N", title="Operational Triage Tier", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Count:Q", title="Inspected Units / Frames"),
        color=alt.Color("Pipeline:N", scale=alt.Scale(domain=pipeline_domain, range=pipeline_colors), legend=alt.Legend(title="Recorded scheduler")),
        xOffset="Pipeline:N",
        tooltip=["Pipeline", "Tier", "Count"]
    ).properties(height=260)
    if chart_data:
        st.altair_chart(action_chart, width="stretch", key=f"{key_prefix}-action-chart")
    else:
        st.info("No recorded triage actions are available to chart.")

    # ── Interactive Records Explorer ──
    with st.expander("Explore benchmark records"):
        display_cols = [c for c in ["image_path", "category", "risk_level", "selected_action", "ppo_final_action", "anomaly_score", "mahalanobis_prediction", "final_decision_status"] if c in table.columns]
        st.dataframe(
            table[display_cols], width="stretch", hide_index=True,
            key=f"{key_prefix}-record-table",
        )
