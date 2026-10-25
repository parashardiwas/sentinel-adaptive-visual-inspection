from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
import html as html_lib
from pathlib import Path

import altair as alt
import streamlit as st

try:
    import cv2
except ImportError:
    st.error(
        "**OpenCV (`cv2`) is missing from the active Python runtime.**\n\n"
        "Your virtual environment `.venv` already has OpenCV installed. Please run:\n\n"
        "```bash\n"
        "source .venv/bin/activate\n"
        "streamlit run app.py\n"
        "```\n"
        "or run directly:\n\n"
        "```bash\n"
        "./.venv/bin/streamlit run app.py\n"
        "```"
    )
    st.stop()
import numpy as np
import pandas as pd
from PIL import Image

from inference2 import (
    ANOMALY_DECISION_THRESHOLD, BUILTIN_CATEGORY_THRESHOLDS, CATEGORIES, CATEGORY_DETECTORS,
    DEFAULT_THRESHOLD_SELECTION_METHOD, DEFAULT_THRESHOLD_SELECTION_PARTITION,
    HybridInspectionEngine, blend_heatmap,
)
from rag.dashboard_adapter import (
    render_rag_audit_tab,
    render_copilot_tab,
    render_benchmark_tab,
)

st.set_page_config(
    page_title="SENTINEL // Command Center",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------- Look & feel -------------------------
inspection_run_status = "RESULT AVAILABLE" if st.session_state.get("sentinel_result") else "AWAITING OPERATOR INPUT"
st.markdown(
    """
    <style>
      :root { color-scheme:dark; --primary-color:#69c9d4; --ink:#e5edf2; --muted:#91a0aa; --cyan:#69c9d4; --blue:#8cb9cf; --line:#263640; --panel:#101a20; }
      html, body, [class*="css"] { font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
      .stApp { background:#0a1116; color:var(--ink); }
      [data-testid="stAppViewContainer"] > .main { position:relative; }
      [data-testid="stMainBlockContainer"] { max-width:1480px; padding-top:1.4rem; padding-bottom:3rem; }
      [data-testid="stHeader"] { background:#0a1116; border-bottom:1px solid #1d2a32; }
      [data-testid="stSidebar"] { background:#0d151b; border-right:1px solid #26343d; }
      [data-testid="stSidebar"] > div:first-child { background:transparent; }
      [data-testid="stSidebar"] h2 { color:#dce6eb; letter-spacing:.04em; text-transform:uppercase; font-size:.82rem; }
      .hero { display:flex; align-items:center; justify-content:space-between; gap:24px; border:1px solid #263740; border-left:3px solid var(--cyan); border-radius:0; padding:18px 24px; background:#101a20; margin:0 0 20px; }
      .eyebrow,.section-kicker,.mono,.hero-meta,.hero-status,.hero-node-label { font-family:'IBM Plex Mono',monospace; }
      .eyebrow { letter-spacing:.18em; text-transform:uppercase; color:var(--cyan); font-size:.7rem; }
      .hero h1 { font-size:1.7rem; line-height:1.1; margin:6px 0 3px; color:#f0f4f6; font-weight:700; letter-spacing:.025em; white-space:nowrap; }
      .hero-subtitle { color:#99abb5; font-family:'IBM Plex Mono',monospace; letter-spacing:.06em; font-size:.72rem; text-transform:uppercase; }
      .hero p { color:#aab7be; max-width:680px; font-size:.88rem; line-height:1.55; margin:10px 0 0; }
      .hero-meta { display:flex; flex-wrap:wrap; justify-content:flex-end; gap:8px 18px; color:#aab7be; font-size:.65rem; letter-spacing:.06em; text-transform:uppercase; }
      .hero-meta span:before { content:''; display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--cyan); margin:0 8px 1px 0; }
      .section-kicker { color:#91aab5; text-transform:uppercase; letter-spacing:.10em; font-size:.67rem; margin-bottom:8px; }
      .panel { border:1px solid var(--line); border-radius:0; background:var(--panel); padding:16px 18px; box-shadow:none; }
      .muted { color:var(--muted); font-size:.84rem; }
      .metric-card { border:1px solid var(--line); border-radius:0; padding:15px 17px; background:#101a20; min-height:100px; }
      .metric-label { color:#8ba7bb; font-size:.76rem; }
      .metric-value { color:#effaff; font-weight:800; font-size:1.75rem; letter-spacing:-.04em; margin-top:4px; }
      div[data-testid="stFileUploader"] { border:1px dashed #40545e; border-radius:0; background:#0d171d; padding:8px; }
      .stButton>button[kind="primary"] { border:1px solid #4d9ba5; border-radius:0; background:#17414a; color:#edf6f7; font-family:'IBM Plex Mono',monospace; letter-spacing:.04em; font-weight:600; min-height:2.65rem; box-shadow:none; }
      .stButton>button[kind="primary"]:hover { border-color:#79cbd2; background:#1c5059; box-shadow:none; }
      .stButton>button:not([kind="primary"]),.stDownloadButton>button { border-radius:0; border-color:#35464f; background:#111c22; color:#c5d0d5; font-family:'IBM Plex Mono',monospace; }
      [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:0; border-bottom:1px solid #2b3b44; }
      .stApp .react-aria-SelectionIndicator { background-color:var(--cyan) !important; }
      [data-testid="stTabs"] button[data-baseweb="tab"] { border:0; border-bottom:2px solid transparent; border-radius:0; color:#9cabb3; font-family:'IBM Plex Mono',monospace; font-size:.69rem; letter-spacing:.025em; padding:11px 16px; }
      [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] { color:#eef4f5 !important; background:#121e24 !important; border-color:transparent !important; border-bottom-color:var(--cyan) !important; box-shadow:none !important; }
      [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"]:after { background:var(--cyan) !important; }
      [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:#eef4f5 !important; box-shadow:none !important; }
      [data-testid="stTabs"] button[data-baseweb="tab"] [data-testid="stMarkdownContainer"] p { white-space:nowrap; }
      [data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid var(--line); border-radius:0; }
      [data-testid="stAlert"] { background:#111d23 !important; border:1px solid #354b55; border-radius:0; color:#dce8ec; }
      [data-testid="stAlert"] > div { background:#111d23 !important; }
      [data-testid="stAlert"] [data-testid="stMarkdownContainer"] { color:#dce8ec; }
      input[type="radio"],input[type="checkbox"] { accent-color:var(--cyan); }
      input,textarea,[data-baseweb="select"] > div { border-radius:2px !important; }
      code,.mono { font-family:'IBM Plex Mono',monospace; }
      hr { border-color:#263740; }
      @media (max-width:760px) { .hero { align-items:flex-start; flex-direction:column; padding:16px; } .hero-meta { justify-content:flex-start; } .hero h1 { font-size:1.55rem; } [data-testid="stTabs"] [data-baseweb="tab-list"] { overflow-x:auto; } }
      @media (prefers-reduced-motion:reduce) { *,*:before,*:after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <section class="hero" aria-label="Sentinel command center">
      <div class="hero-copy">
        <div class="eyebrow">Adaptive Industrial Intelligence</div>
        <h1>SENTINEL</h1>
        <div class="hero-subtitle">// Command Center</div>
      </div>
      <div class="hero-meta"><span>{inspection_run_status}</span><span>Local execution</span></div>
    </section>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Sidebar controls -------------------------
st.sidebar.markdown("## Inspection setup")
default_project = os.environ.get("MVTEC_PROJECT_DIR", str(Path.cwd()))
project_dir = st.sidebar.text_input(
    "Model project folder",
    value=default_project,
    help="Folder containing models/ and assets/ exported from your Google Drive project.",
)
device_option = st.sidebar.selectbox("Compute device", ["auto", "cuda", "cpu"], index=0)

PIPELINE_SUITES = [
    "Pipeline B: Deep visual inspection + PPO",
    "Pipeline A: ResNet-18 + Mahalanobis + Q-Learning",
    "Benchmark Mode: Side-by-Side Comparison"
]

if "active_architecture_suite" not in st.session_state:
    st.session_state["active_architecture_suite"] = PIPELINE_SUITES[0]
if "sidebar_suite_select" not in st.session_state:
    st.session_state["sidebar_suite_select"] = st.session_state["active_architecture_suite"]

def _sync_suite_from_sidebar():
    st.session_state["active_architecture_suite"] = st.session_state["sidebar_suite_select"]

st.sidebar.selectbox(
    "Architecture Suite",
    PIPELINE_SUITES,
    key="sidebar_suite_select",
    on_change=_sync_suite_from_sidebar,
    help="Choose whether to run Pipeline A, Pipeline B, or evaluate both side-by-side."
)

model_suite = st.session_state["active_architecture_suite"]
is_pipeline_a = "Pipeline A" in model_suite
is_pipeline_b = "Pipeline B" in model_suite
is_benchmark_mode = "Benchmark" in model_suite
inspection_mode = st.sidebar.radio(
    "Inspection mode",
    ["Anomaly detection only", "RL-assisted inspection"],
    index=1,
    help="Anomaly detection only runs the visual detector and heatmaps. RL-assisted inspection additionally runs the temporal model and trained PPO policy.",
)
enable_rl = inspection_mode == "RL-assisted inspection"
category = st.sidebar.selectbox("Object category", CATEGORIES, index=0, help="Choose the object type being inspected. The hybrid model route depends on this choice.")
# Use the category thresholds from the last controller_dev F2 calibration run.
# The JSON generated by Colab overrides the bundled snapshot when present.
threshold_config_path = Path(project_dir).expanduser() / "assets" / "decision_thresholds.json"
threshold_config = {}
threshold_file_read_ok = False
try:
    if threshold_config_path.is_file():
        with threshold_config_path.open("r", encoding="utf-8") as _tf:
            threshold_config = json.load(_tf)
        threshold_file_read_ok = True
except Exception as _threshold_exc:
    st.sidebar.error(f"Could not read category decision thresholds: {_threshold_exc}")
    st.sidebar.warning("The inference engine refuses to use a malformed threshold file. Fix or remove decision_thresholds.json; removing it enables the bundled category thresholds.")

_threshold_map = dict(BUILTIN_CATEGORY_THRESHOLDS)
_file_threshold_map = threshold_config.get("thresholds", {}) if isinstance(threshold_config, dict) else {}
_valid_file_threshold_categories = set()
if isinstance(_file_threshold_map, dict):
    for _cat, _value in _file_threshold_map.items():
        if _cat in CATEGORIES:
            try:
                _value = float(_value)
                if np.isfinite(_value) and 0.0 <= _value <= 1.0:
                    _threshold_map[_cat] = _value
                    _valid_file_threshold_categories.add(_cat)
            except (TypeError, ValueError):
                pass

_has_category_threshold = category in _valid_file_threshold_categories
decision_boundary = float(_threshold_map.get(category, ANOMALY_DECISION_THRESHOLD))
if _has_category_threshold:
    _selection_method = threshold_config.get("selection_method", threshold_config.get("policy", DEFAULT_THRESHOLD_SELECTION_METHOD))
    _threshold_description = "loaded from assets/decision_thresholds.json"
    _threshold_details = f"Selected on `{threshold_config.get('selected_on', DEFAULT_THRESHOLD_SELECTION_PARTITION)}`."
elif category in BUILTIN_CATEGORY_THRESHOLDS:
    _selection_method = DEFAULT_THRESHOLD_SELECTION_METHOD
    _threshold_description = "bundled calibration values"
    _threshold_details = "These are category-specific thresholds selected on controller_dev in the latest calibration run. Copy the JSON from Drive to override them with your saved config."
else:
    _selection_method = "legacy compatibility fallback"
    _threshold_description = "legacy 0.95 fallback"
    _threshold_details = "No category-specific threshold is available."
st.sidebar.success(
    f"**Category decision policy**\n\n**{category} cutoff: {decision_boundary:.4f}**\n\n"
    f"Source: {_threshold_description}. {_threshold_details}\n\n"
    "The score stays continuous; this cutoff is used only for the ANOMALY/NORMAL label."
)
if not threshold_file_read_ok:
    st.sidebar.caption("The selected per-category thresholds are bundled in this app version. A valid assets/decision_thresholds.json overrides them.")
sampling_interval = st.sidebar.select_slider(
    "Sample one frame every",
    options=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
    value=0.25,
    format_func=lambda v: f"{v:g} sec",
    help="PPO V4 was trained on synthetic sequences at 4 decisions/second. 0.25 seconds best matches that cadence when the frame budget allows it.",
)
max_frames = st.sidebar.slider("Maximum sampled frames", min_value=24, max_value=360, value=180, step=12)
enable_subtypes = st.sidebar.checkbox("Try defect-subtype classification", value=False, help="Optional DINOv2 + LinearSVM output for frames with high anomaly scores. Requires the saved DINOv2 backbone and category classifier; it may download DINOv2 source code on first use.")

st.sidebar.markdown("### Product region / ROI")
roi_mode = st.sidebar.radio(
    "Region sent to the detector",
    ["Center square crop (recommended)", "Full frame", "Custom ROI"],
    index=0,
    help="MVTec training images are square and tightly framed. Real video often includes background, borders, perspective, or empty space; cropping the product region can reduce domain-shift false alarms.",
)
roi_x_pct, roi_y_pct, roi_w_pct, roi_h_pct = 0, 0, 100, 100
if roi_mode == "Custom ROI":
    roi_x_pct = st.sidebar.slider("ROI left (%)", 0, 95, 10, 1)
    roi_y_pct = st.sidebar.slider("ROI top (%)", 0, 95, 10, 1)
    roi_w_pct = st.sidebar.slider("ROI width (%)", 5, 100, 80, 1)
    roi_h_pct = st.sidebar.slider("ROI height (%)", 5, 100, 80, 1)

st.sidebar.caption("If most frames score near 1.0, check this ROI, the selected category, camera lighting, and object scale before trusting the alerts.")
st.sidebar.markdown("---")
st.sidebar.markdown("---")
st.sidebar.markdown("<div class='muted'>Recommended: use a GPU for long videos. Video inference runs locally on the machine hosting this app.</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='muted'>MVTec-trained research prototype, not a certified safety inspection system.</div>", unsafe_allow_html=True)

# ------------------------- Model readiness -------------------------
project_path = Path(project_dir).expanduser()
models_path = project_path / "models"
assets_path = project_path / "assets"
base_requirements = [
    models_path / "yolo26n.pt",
    models_path / "wide_resnet50_2_patch_extractor.pth",
    models_path / "stfpm_teacher_resnet18_grid.pth",
    models_path / "stfpm_student_resnet18_grid.pth",
    assets_path / "stfpm_calibration_grid.npz",
]
if enable_rl:
    base_requirements.extend([
        models_path / "causal_temporal_transformer.pt",
        models_path / "temporal_transformer_input_scaler.joblib",
    ])
    v4_policy_file = models_path / "ppo_monitor_flag_escalate_recall_v4toothbrush.zip"
    legacy_policy_file = models_path / "ppo_inspection_policy.zip"
    # Note: if PPO zip is not present, system runs adaptive triage rather than blocking startup.
for cat in CATEGORIES:
    base_requirements.extend([
        models_path / f"yolo26_embedding_knn_{cat}.joblib",
        models_path / f"patchcore_patch_bank_{cat}.pt",
        assets_path / f"calibration_{cat}.npz",
    ])
missing_paths = [p for p in base_requirements if not p.is_file()]

if missing_paths:
    st.warning(
        "The model bundle is incomplete at the selected project folder. Export your trained `models/` and `assets/` folders from Google Drive, then select the folder that contains those two subfolders."
    )
    with st.expander("Missing required artifacts"):
        for missing_path in missing_paths:
            st.code(str(missing_path), language="text")
    st.info("The model project path uses MVTEC_PROJECT_DIR when set; otherwise it defaults to the current working directory.")
    engine = None
else:
    def _threshold_file_signature(path: Path) -> str:
        try:
            stat = path.stat()
            return f"{stat.st_mtime_ns}:{stat.st_size}"
        except OSError:
            return "missing"

    @st.cache_resource(show_spinner="Loading the visual detector and selected inspection route…")
    def load_engine_cached(path: str, device: str, use_rl: bool, threshold_signature: str):
        # threshold_signature is intentionally part of the cache key so changing the JSON
        # cannot leave an old threshold map inside a cached inference engine.
        return HybridInspectionEngine(path, device=device, enable_rl=use_rl)

# Controller mode detection without eagerly loading 50+ model weights on initial page render
    v4_policy = models_path / "ppo_monitor_flag_escalate_recall_v4toothbrush.zip"
    legacy_policy = models_path / "ppo_inspection_policy.zip"

    if is_pipeline_a:
        st.warning("Pipeline A live inference is unavailable in this checkout: the UI has no exported Mahalanobis model or Q-table. Pipeline A results are disabled rather than synthesized.")


# ------------------------- Primary Navigation Tabs -------------------------

def resize_for_display(frame: np.ndarray, max_w=720, max_h=480) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, max_w / max(w, 1), max_h / max(h, 1))
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def compute_roi_bbox(frame_shape, mode: str, x_pct=0, y_pct=0, w_pct=100, h_pct=100):
    """Return (x1,y1,x2,y2) in source-frame pixels; end coordinates are exclusive."""
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if h < 2 or w < 2:
        return (0, 0, max(w, 1), max(h, 1))
    if mode == "Center square crop (recommended)":
        side = min(h, w)
        x1 = (w - side) // 2
        y1 = (h - side) // 2
        return (x1, y1, x1 + side, y1 + side)
    if mode == "Custom ROI":
        x1 = int(round(w * float(x_pct) / 100.0))
        y1 = int(round(h * float(y_pct) / 100.0))
        x2 = int(round(w * (float(x_pct) + float(w_pct)) / 100.0))
        y2 = int(round(h * (float(y_pct) + float(h_pct)) / 100.0))
        x1 = min(max(x1, 0), w - 1)
        y1 = min(max(y1, 0), h - 1)
        x2 = min(max(x2, x1 + 1), w)
        y2 = min(max(y2, y1 + 1), h)
        return (x1, y1, x2, y2)
    return (0, 0, w, h)


def blend_heatmap_in_roi(display_frame, heatmap, source_bbox, source_shape, alpha=0.48):
    """Map a crop-space heatmap back into the matching ROI of the display frame only."""
    display = display_frame.copy()
    dh, dw = display.shape[:2]
    sh, sw = int(source_shape[0]), int(source_shape[1])
    x1, y1, x2, y2 = source_bbox
    sx = dw / max(sw, 1)
    sy = dh / max(sh, 1)
    dx1 = int(round(x1 * sx)); dx2 = int(round(x2 * sx))
    dy1 = int(round(y1 * sy)); dy2 = int(round(y2 * sy))
    dx1 = min(max(dx1, 0), dw - 1); dx2 = min(max(dx2, dx1 + 1), dw)
    dy1 = min(max(dy1, 0), dh - 1); dy2 = min(max(dy2, dy1 + 1), dh)
    roi = display[dy1:dy2, dx1:dx2]
    heat_overlay = blend_heatmap(roi, heatmap, alpha=alpha)
    display[dy1:dy2, dx1:dx2] = heat_overlay
    # Explicitly show the evaluated region so users know background outside it was not scored.
    cv2.rectangle(display, (dx1, dy1), (dx2 - 1, dy2 - 1), (210, 230, 100), 1, cv2.LINE_AA)
    return display


def make_annotated_panel(
    original_bgr: np.ndarray,
    heat_bgr: np.ndarray,
    row: dict,
    category_name: str,
    inspection_mode: str,
    score_history: list[float],
    decision_boundary: float,
) -> np.ndarray:
    """Render source frame, the existing ROI-aligned heatmap, action/score and a score sparkline."""
    panel_w, panel_h = 420, 260

    def fit(img):
        return cv2.resize(img, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

    original = fit(original_bgr)
    heat = fit(heat_bgr)
    info = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    info[:] = (25, 31, 48)
    score = float(row.get("anomaly_score", 0.0))
    is_rl = inspection_mode == "RL-assisted inspection"
    action = str(row.get("ppo_final_action", "DETECTION ONLY")) if is_rl else "DETECTION ONLY"
    tfp = row.get("temporal_anomaly_probability", np.nan)
    delta = float(row.get("score_delta", 0.0))
    if action == "HUMAN_REVIEW":
        action_color = (70, 80, 255)
    elif action == "FLAG_ANOMALY":
        action_color = (20, 170, 255)
    elif action == "MONITOR":
        action_color = (95, 210, 155)
    else:
        action_color = (190, 210, 220)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(info, "INSPECTION SUMMARY", (18, 28), font, 0.58, (160, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(info, f"{category_name.upper()}  |  {str(row.get('detector', ''))[:27]}", (18, 55), font, 0.48, (235, 240, 250), 1, cv2.LINE_AA)
    cv2.putText(info, f"ANOMALY SCORE   {score:.3f}", (18, 92), font, 0.68, (70, 230, 205), 2, cv2.LINE_AA)
    prediction = str(row.get("detector_prediction", "ANOMALY" if row.get("is_anomaly", score >= decision_boundary) else "NORMAL"))
    prediction_color = (55, 65, 255) if prediction == "ANOMALY" else (80, 220, 145)
    cv2.putText(info, f"MODEL PREDICTION: {prediction}", (18, 119), font, 0.51, prediction_color, 2, cv2.LINE_AA)
    if is_rl and pd.notna(tfp):
        cv2.putText(info, f"Temporal {float(tfp):.3f}  |  score delta {delta:+.3f}", (18, 141), font, 0.42, (225, 230, 240), 1, cv2.LINE_AA)
    cv2.putText(info, action[:28], (18, 164), font, 0.60 if len(action) < 18 else 0.46, action_color, 2, cv2.LINE_AA)

    # In-video score trace; the notebook decision boundary is fixed and shown for context.
    graph_x, graph_y, graph_w, graph_h = 20, 190, 380, 42
    cv2.putText(info, f"ANOMALY SCORE · DECISION LINE {decision_boundary:.3f}", (graph_x, graph_y - 5), font, 0.37, (160, 185, 205), 1, cv2.LINE_AA)
    cv2.rectangle(info, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (75, 88, 110), 1, cv2.LINE_AA)
    threshold_y = graph_y + graph_h - int(np.clip(decision_boundary, 0, 1) * (graph_h - 8)) - 4
    cv2.line(info, (graph_x + 1, threshold_y), (graph_x + graph_w - 1, threshold_y), (50, 120, 225), 1, cv2.LINE_AA)
    values = np.asarray(score_history[-60:], dtype=np.float32)
    if values.size >= 1:
        points = []
        for i, val in enumerate(values):
            x = graph_x + 3 + int(i * (graph_w - 6) / max(len(values) - 1, 1))
            y = graph_y + graph_h - 4 - int(np.clip(float(val), 0, 1) * (graph_h - 8))
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(info, [np.asarray(points, dtype=np.int32)], False, (80, 230, 185), 2, cv2.LINE_AA)
        else:
            cv2.circle(info, points[0], 3, (80, 230, 185), -1, cv2.LINE_AA)
    combined = np.hstack([original, heat, info])
    header = np.zeros((44, combined.shape[1], 3), dtype=np.uint8)
    header[:] = (10, 16, 30)
    cv2.putText(
        header,
        f"SENTINEL / FRAME {row.get('frame_index', '')} / t={float(row.get('timestamp_sec', 0)):.2f}s",
        (16, 29), font, 0.60, (245, 248, 255), 1, cv2.LINE_AA,
    )
    footer = np.zeros((34, combined.shape[1], 3), dtype=np.uint8)
    footer[:] = (10, 16, 30)
    footer_text = "LOCALIZATION MAP IS FRAME-NORMALIZED · SCORE IS A NORMAL-REFERENCE PERCENTILE"
    if is_rl:
        footer_text = "PPO ACTIONS ARE EXPERIMENTAL TRIAGE · HUMAN CONFIRMATION REMAINS IMPORTANT"
    cv2.putText(footer, footer_text, (16, 22), font, 0.45, (150, 175, 195), 1, cv2.LINE_AA)
    return np.vstack([header, combined, footer])


def make_browser_compatible_video(source_path: Path, output_dir: Path):
    """Transcode OpenCV's mp4v output to browser-safe H.264 MP4 (or VP9 WebM fallback)."""
    ffmpeg = shutil.which("ffmpeg")
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    if ffmpeg:
        mp4_path = output_dir / "sentinel_annotated_inspection.mp4"
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_path),
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if completed.returncode == 0 and mp4_path.is_file() and mp4_path.stat().st_size > 1000:
                return mp4_path, "video/mp4", None
        except Exception:
            pass
        webm_path = output_dir / "sentinel_annotated_inspection.webm"
        webm_cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_path),
            "-an", "-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "5",
            "-b:v", "0", "-crf", "31", str(webm_path),
        ]
        try:
            completed = subprocess.run(webm_cmd, capture_output=True, text=True, timeout=180)
            if completed.returncode == 0 and webm_path.is_file() and webm_path.stat().st_size > 1000:
                return webm_path, "video/webm", "H.264 export was unavailable, so browser playback uses WebM/VP9."
        except Exception:
            pass
    return source_path, "video/mp4", "Could not transcode this OpenCV MP4 to a browser-friendly codec. The raw MP4 is still downloadable; install imageio-ffmpeg and rerun to enable H.264 playback."


@st.cache_data(show_spinner=False, max_entries=4)
def cached_browser_video(source_bytes: bytes, suffix: str):
    """Create a browser-playable preview once per uploaded video, reusing it across reruns."""
    with tempfile.TemporaryDirectory(prefix="sentinel_video_cache_") as temp_dir:
        temp_root = Path(temp_dir)
        source_path = temp_root / f"source{suffix or '.mp4'}"
        source_path.write_bytes(source_bytes)
        output_dir = temp_root / "browser"
        output_dir.mkdir()
        browser_path, mime, warning = make_browser_compatible_video(source_path, output_dir)
        return browser_path.read_bytes(), mime, warning


def svg_score_chart(table: pd.DataFrame, decision_boundary: float, width: int = 980, height: int = 260) -> str:
    """Small dependency-free score chart embedded in downloadable HTML reports."""
    values = pd.to_numeric(table.get("anomaly_score", pd.Series(dtype=float)), errors="coerce").fillna(0).to_numpy()
    if values.size == 0:
        return "<p>No score samples available.</p>"
    left, right, top, bottom = 52, width - 22, 20, height - 36
    usable_w, usable_h = right - left, bottom - top
    points = []
    for i, value in enumerate(values):
        x = left + i * usable_w / max(len(values) - 1, 1)
        y = bottom - float(np.clip(value, 0, 1)) * usable_h
        points.append(f"{x:.1f},{y:.1f}")
    t_y = bottom - float(np.clip(decision_boundary, 0, 1)) * usable_h
    grid = []
    for tick in (0.0, 0.25, 0.50, 0.75, 1.0):
        gy = bottom - tick * usable_h
        grid.append(
            f'<line x1="{left}" y1="{gy:.1f}" x2="{right}" y2="{gy:.1f}" stroke="#26374f" stroke-width="1"/>'
            f'<text x="8" y="{gy + 4:.1f}" fill="#9fb0c7" font-size="11">{tick:.2f}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Anomaly score timeline with fixed zero-to-one axis" style="width:100%;height:auto">'
        f'<rect x="{left}" y="{top}" width="{usable_w}" height="{usable_h}" rx="8" fill="#101a2b" stroke="#2a3850"/>'
        + "".join(grid) +
        f'<line x1="{left}" y1="{t_y:.1f}" x2="{right}" y2="{t_y:.1f}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="6 5"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#5eead4" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'
        f'<text x="{left}" y="{height - 8}" fill="#9fb0c7" font-size="12">first sampled frame</text>'
        f'<text x="{right - 130}" y="{height - 8}" fill="#9fb0c7" font-size="12">last sampled frame</text>'
        f'<text x="{right - 160}" y="{max(top + 13, t_y - 7):.1f}" fill="#fbbf24" font-size="12">threshold {decision_boundary:.4f}</text></svg>'
    )


def build_dashboard_html(report: dict, table: pd.DataFrame) -> bytes:
    """Build a portable, standalone HTML report/dashboard with summary, timeline and top frames."""
    def esc(value):
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "—"
        return html_lib.escape(str(value))

    decision_boundary = float(report.get("decision_boundary", ANOMALY_DECISION_THRESHOLD))
    high_n = int(report.get("high_anomaly_frame_count", 0))
    n = int(report.get("sampled_frames", len(table)))
    max_row = report.get("max_score_location", {})
    min_row = report.get("min_score_location", {})
    is_rl = report.get("inspection_mode") == "RL-assisted inspection"
    metrics = [
        ("Frames sampled", f"{n:,}"),
        (f"Model-predicted anomalies", f"{high_n:,}"),
        ("Mean score", f"{float(report.get('mean_anomaly_score', 0)):.3f}"),
        ("Peak score", f"{float(report.get('max_anomaly_score', 0)):.3f}"),
        ("Lowest score", f"{float(report.get('min_anomaly_score', 0)):.3f}"),
    ]
    if is_rl:
        metrics += [("Human-review events", f"{int(report.get('human_review_event_count', 0)):,}"),
                    ("Frames labelled FLAG_ANOMALY", f"{int(report.get('flag_action_frames', 0)):,}")]
    cards = "".join(f'<div class="metric"><div class="label">{esc(label)}</div><div class="value">{esc(value)}</div></div>' for label, value in metrics)
    chart = svg_score_chart(table, decision_boundary)

    cols = [c for c in ["frame_index", "timestamp_sec", "anomaly_score", "detector_prediction", "is_anomaly", "ppo_final_action", "human_review_event"] if c in table.columns]
    if "anomaly_score" in table.columns:
        top = table.sort_values("anomaly_score", ascending=False).head(12)
    else:
        top = table.head(0)
    def rows_html(df, columns):
        body = []
        for _, rr in df[columns].iterrows():
            cells = []
            for col in columns:
                val = rr[col]
                if isinstance(val, (float, np.floating)) and np.isfinite(val):
                    if col in {"timestamp_sec"}:
                        val = f"{float(val):.2f}s"
                    elif col in {"anomaly_score"}:
                        val = f"{float(val):.3f}"
                    else:
                        val = f"{float(val):.3f}"
                cells.append(f"<td>{esc(val)}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return "".join(body) or f'<tr><td colspan="{max(1,len(columns))}">No rows</td></tr>'
    top_table = "<table><thead><tr>" + "".join(f"<th>{esc(c.replace('_',' ').title())}</th>" for c in cols) + "</tr></thead><tbody>" + rows_html(top, cols) + "</tbody></table>"
    action_section = ""
    if is_rl:
        counts = report.get("ppo_action_counts", {}) or {}
        bars = []
        max_count = max(counts.values(), default=1)
        for name in ["MONITOR", "FLAG_ANOMALY", "HUMAN_REVIEW", "DO_NOTHING"]:
            count = int(counts.get(name, 0))
            if name not in counts and count == 0:
                continue
            width_pct = 0 if max_count == 0 else (count / max_count * 100)
            bars.append(f'<div class="barrow"><div class="barlabel">{esc(name)}</div><div class="bartrack"><div class="bar" style="width:{width_pct:.1f}%"></div></div><div class="barvalue">{count:,}</div></div>')
        reviews = table[table.get("ppo_final_action", pd.Series(index=table.index, dtype=str)).astype(str).eq("HUMAN_REVIEW")] if "ppo_final_action" in table.columns else table.head(0)
        review_cols = [c for c in ["frame_index", "timestamp_sec", "anomaly_score", "human_review_event"] if c in reviews.columns]
        review_table = "<table><thead><tr>" + "".join(f"<th>{esc(c.replace('_',' ').title())}</th>" for c in review_cols) + "</tr></thead><tbody>" + rows_html(reviews.head(30), review_cols) + "</tbody></table>"
        action_section = f'<section><h2>RL action mix</h2>{"".join(bars) or "<p>No action counts available.</p>"}<h2>Human-review frames</h2>{review_table}</section>'

    caveat = esc(report.get("research_caveat", ""))
    meta = f"{esc(report.get('video_name','uploaded video'))} · {esc(report.get('category',''))} · {esc(report.get('inspection_mode',''))} · {esc(report.get('detector',''))}"
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentinel Inspection Dashboard</title>
<style>
:root{{--bg:#0b1020;--panel:#111a2d;--stroke:#2a3850;--text:#edf3ff;--muted:#9fb0c7;--mint:#5eead4;--amber:#fbbf24}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(ellipse at 10% 0%,#123039 0,transparent 35%),var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}main{{max-width:1200px;margin:auto;padding:32px 24px 60px}}.eyebrow{{font:12px ui-monospace,monospace;letter-spacing:.15em;text-transform:uppercase;color:var(--mint)}}h1{{font-size:clamp(32px,5vw,54px);line-height:1.04;letter-spacing:-.04em;margin:10px 0}}.meta{{color:var(--muted);margin-bottom:26px}}.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:12px;margin:22px 0}}.metric,section{{border:1px solid var(--stroke);background:rgba(17,26,45,.88);border-radius:16px;padding:18px}}.label{{color:var(--muted);font-size:13px}}.value{{font-size:28px;font-weight:800;color:#fff;margin-top:5px}}section{{margin-top:18px}}h2{{font-size:20px;margin:0 0 16px}}p,.note{{color:var(--muted)}}table{{width:100%;border-collapse:collapse;font-size:13px;overflow-wrap:anywhere}}th,td{{padding:10px 9px;border-bottom:1px solid var(--stroke);text-align:left}}th{{color:var(--mint);font-weight:700}}.tablewrap{{overflow-x:auto}}.barrow{{display:grid;grid-template-columns:170px 1fr 55px;gap:12px;align-items:center;margin:14px 0}}.barlabel{{font:12px ui-monospace,monospace}}.bartrack{{background:#0b1020;border-radius:99px;height:13px;overflow:hidden;border:1px solid var(--stroke)}}.bar{{height:100%;background:linear-gradient(90deg,#0ea5a4,#6366f1);border-radius:99px}}.barvalue{{text-align:right;font-weight:700}}.foot{{font-size:12px;color:var(--muted);margin-top:24px}}@media(max-width:600px){{.barrow{{grid-template-columns:120px 1fr 40px}}main{{padding:20px 12px}}}}
</style></head><body><main><div class="eyebrow">Sentinel · Inspection intelligence</div><h1>Inspection dashboard<span style="color:var(--mint)">.</span></h1><div class="meta">{meta}</div><div class="metrics">{cards}</div>
<section><h2>Anomaly score timeline</h2>{chart}<p>Dashed amber line = this category’s decision cutoff. The y-axis is fixed from 0 to 1. The score is a normal-reference percentile, not a probability of defect.</p><p>{int(report.get('score_saturated_at_one_count', 0))} of {int(report.get('sampled_frames', 0))} sampled frames ({100.0 * float(report.get('score_saturated_at_one_rate', 0.0)):.1f}%) reached 1.000. ECDF calibration maps a score at or above the largest normal-reference score to 1.000; see the exported frame-level CSV’s pre_final_calibration_score column for the values before that mapping.</p></section>
<section><h2>Peak / minimum score location</h2><div class="tablewrap"><table><thead><tr><th>Statistic</th><th>Score</th><th>Source frame</th><th>Timestamp</th></tr></thead><tbody><tr><td>Maximum</td><td>{float(report.get('max_anomaly_score',0)):.3f}</td><td>{esc(max_row.get('frame_index','—'))}</td><td>{esc(max_row.get('timestamp_sec','—'))}</td></tr><tr><td>Minimum</td><td>{float(report.get('min_anomaly_score',0)):.3f}</td><td>{esc(min_row.get('frame_index','—'))}</td><td>{esc(min_row.get('timestamp_sec','—'))}</td></tr></tbody></table></div></section>
<section><h2>Highest-scoring sampled frames</h2><div class="tablewrap">{top_table}</div></section>
{action_section}
<section><h2>Interpretation & limitation</h2><p>{caveat}</p><p>Review the source frame and heatmap alongside the numeric score. A high score means unusual relative to the saved normal calibration reference; it does not prove a defect.</p></section>
<div class="foot">Generated by Sentinel · Local inference · Exported dashboard · {esc(report.get('video_name',''))}</div></main></body></html>"""
    return html.encode("utf-8")

def process_video_file(video_bytes: bytes, original_name: str, active_engine: HybridInspectionEngine,
                       active_category: str, interval_sec: float, frame_limit: int,
                       subtype_predictions: bool, roi_mode: str = "Center square crop (recommended)",
                       roi_x_pct: int = 0, roi_y_pct: int = 0,
                       roi_w_pct: int = 100, roi_h_pct: int = 100,
                       inspection_mode: str = "RL-assisted inspection",
                       decision_boundary: float = ANOMALY_DECISION_THRESHOLD):
    workdir = Path(tempfile.mkdtemp(prefix="sentinel_inspection_"))
    video_path = workdir / (Path(original_name).stem + Path(original_name).suffix.lower())
    video_path.write_bytes(video_bytes)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open this video. Try MP4 (H.264) or AVI.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps if frame_count > 0 else None
    stride = max(1, int(round(fps * interval_sec)))
    if frame_count > 0:
        stride = max(stride, int(np.ceil(frame_count / frame_limit)))
    effective_interval = stride / fps
    if inspection_mode == "RL-assisted inspection" and getattr(active_engine, "ppo_mode", "") == "v4_recall_balanced" and abs(effective_interval - 0.25) > 0.05:
        st.warning(
            f"PPO V4 was trained at 4 synthetic decisions/second; the effective sampled cadence here is {effective_interval:.2f}s because of the video length/frame budget. Treat temporal decisions as experimental."
        )
    rows = []
    meta = []
    frame_idx = 0
    sample_idx = 0
    progress = st.progress(0, text="Reading video frames…")
    status = st.empty()
    try:
        while sample_idx < frame_limit:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0 and sample_idx < frame_limit:
                status.caption(f"Scoring sampled frame {sample_idx + 1} · source frame {frame_idx}")
                original_shape = frame.shape[:2]
                roi_bbox = compute_roi_bbox(
                    original_shape, roi_mode,
                    x_pct=roi_x_pct, y_pct=roi_y_pct,
                    w_pct=roi_w_pct, h_pct=roi_h_pct,
                )
                rx1, ry1, rx2, ry2 = roi_bbox
                frame_for_model = frame[ry1:ry2, rx1:rx2]
                if frame_for_model.size == 0 or min(frame_for_model.shape[:2]) < 16:
                    raise RuntimeError(
                        f"ROI is too small at source frame {frame_idx}: {roi_bbox}. "
                        "Increase the ROI width/height or use center/full frame."
                    )
                # Score only the selected object region; map the heatmap back to that
                # same region instead of stretching it over the whole camera frame.
                scored = active_engine.score_frame(frame_for_model, active_category)
                display_frame = resize_for_display(frame, max_w=720, max_h=480)
                heat_overlay = blend_heatmap_in_roi(
                    display_frame, scored["heatmap"], roi_bbox, original_shape
                )
                source_path = workdir / f"frame_{sample_idx:05d}.jpg"
                roi_path = workdir / f"roi_{sample_idx:05d}.jpg"
                heat_path = workdir / f"heat_{sample_idx:05d}.jpg"
                cv2.imwrite(str(source_path), display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                cv2.imwrite(str(roi_path), frame_for_model, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                cv2.imwrite(str(heat_path), heat_overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                row = {
                    "frame_index": int(frame_idx),
                    "sample_index": int(sample_idx),
                    "timestamp_sec": round(frame_idx / fps, 4),
                    "category": active_category,
                    "inspection_mode": inspection_mode,
                    "decision_boundary": float(scored.get("decision_boundary", decision_boundary)),
                    "decision_threshold_source": str(scored.get("decision_threshold_source", "")),
                    "is_anomaly": bool(scored.get("is_anomaly", float(scored["anomaly_score"]) >= decision_boundary)),
                    "detector_prediction": str(scored.get("detector_prediction", "ANOMALY" if float(scored["anomaly_score"]) >= decision_boundary else "NORMAL")),

                    "roi_mode": roi_mode,
                    "roi_x1": int(rx1), "roi_y1": int(ry1),
                    "roi_x2": int(rx2), "roi_y2": int(ry2),
                    "detector": scored["detector"],
                    "raw_score": float(scored["raw_score"]),
                    "pre_final_calibration_score": float(scored.get("pre_final_calibration_score", scored["raw_score"])),
                    "anomaly_score": float(scored["anomaly_score"]),
                    "calibration_reference_count": int(scored.get("calibration_reference_count", 0)),
                    "calibration_reference_max_raw_score": float(scored.get("calibration_reference_max_raw_score", np.nan)),
                    "score_saturated_at_one": bool(scored.get("score_saturated_at_one", float(scored["anomaly_score"]) >= 1.0)),
                    "global_evidence_pct": float(scored["global_evidence_pct"]),
                    "local_evidence_pct": float(scored["local_evidence_pct"]),
                    "evidence_dispersion_pct": float(scored["evidence_dispersion_pct"]),
                    "evidence_max_pct": float(scored["evidence_max_pct"]),
                    "stfpm_or_patch_details": json.dumps(scored.get("details", {})),
                    "source_frame_path": str(source_path),
                    "model_input_path": str(roi_path),
                    "heat_overlay_path": str(heat_path),
                }
                rows.append(row)
                meta.append({"source_shape": original_shape, "workdir": str(workdir)})
                sample_idx += 1
                if frame_count > 0:
                    progress.progress(min(0.70, 0.70 * frame_idx / max(frame_count, 1)), text="Scoring frames…")
            frame_idx += 1
    finally:
        cap.release()
    if not rows:
        shutil.rmtree(workdir, ignore_errors=True)
        raise RuntimeError("No frames were sampled from the uploaded video.")
    if inspection_mode == "RL-assisted inspection":
        status.caption("Building temporal evidence and inspection decisions…")
        policy_label = "PPO V4" if getattr(active_engine, "ppo_mode", "") == "v4_recall_balanced" else "adaptive triage fallback"
        progress.progress(0.75, text=f"Running temporal model + {policy_label}…")
        temporal_rows = active_engine.temporal_and_rl(rows)
    else:
        status.caption("Preparing detection-only results…")
        progress.progress(0.75, text="Preparing detector-only frames…")
        temporal_rows = []
        score_history = []
        for i, row in enumerate(rows):
            item = dict(row)
            score = float(item["anomaly_score"])
            item["score_delta"] = float(score - score_history[-1]) if score_history else 0.0
            item["rolling_mean_score_5"] = float(np.mean((score_history + [score])[-5:]))
            item["rolling_std_score_5"] = float(np.std((score_history + [score])[-5:]))
            item["decision"] = str(item.get("detector_prediction", "ANOMALY" if bool(item.get("is_anomaly", score >= decision_boundary)) else "NORMAL"))
            temporal_rows.append(item)
            score_history.append(score)

    if subtype_predictions:
        status.caption("Running optional subtype classifier on model-predicted anomaly frames…")
        for item in temporal_rows:
            item["predicted_subtype"] = ""
            # Subtype diagnosis follows the detector prediction. PPO action remains a separate
            # decision about what to do next (monitor, flag, or human review).
            if bool(item.get("is_anomaly", float(item["anomaly_score"]) >= decision_boundary)):
                # Use the same cropped ROI that was scored by the anomaly detector.
                img_path = item.get("model_input_path") or item["source_frame_path"]
                img = cv2.imread(img_path)
                if img is None:
                    continue
                try:
                    result = active_engine.predict_subtype(img, active_category)
                    if result:
                        item["predicted_subtype"] = result["predicted_subtype"]
                        item["subtype_decision_margin"] = result.get("decision_margin")
                except Exception as exc:
                    item["predicted_subtype"] = f"classifier unavailable: {str(exc)[:120]}"
                    break

    progress.progress(0.82, text="Rendering annotated inspection video with score timeline…")
    raw_video_path = workdir / "sentinel_annotated_inspection_raw.mp4"
    video_writer = None
    output_fps = float(np.clip(1.0 / max(effective_interval, 1.0 / fps), 0.5, 12.0))
    score_history_for_video = []
    try:
        for index, row in enumerate(temporal_rows):
            frame_bgr = cv2.imread(row["source_frame_path"])
            heat_bgr = cv2.imread(row["heat_overlay_path"])
            if frame_bgr is None or heat_bgr is None:
                continue
            score_history_for_video.append(float(row.get("anomaly_score", 0.0)))
            annotated = make_annotated_panel(
                frame_bgr, heat_bgr, row, active_category,
                inspection_mode=inspection_mode,
                score_history=score_history_for_video,
                decision_boundary=decision_boundary,
            )
            if video_writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(raw_video_path), fourcc, output_fps, (w, h))
                if not video_writer.isOpened():
                    raise RuntimeError("Could not initialize the OpenCV video writer. Try installing a full OpenCV/FFmpeg build.")
            video_writer.write(annotated)
            if index % max(1, len(temporal_rows) // 10) == 0:
                progress.progress(0.82 + 0.10 * (index + 1) / len(temporal_rows), text="Rendering annotated video…")
    finally:
        if video_writer is not None:
            video_writer.release()

    if not raw_video_path.is_file() or raw_video_path.stat().st_size < 1000:
        raise RuntimeError("The annotated video did not render correctly.")

    processed_video_path, video_mime_type, playback_warning = make_browser_compatible_video(raw_video_path, workdir)

    if not processed_video_path.is_file() or processed_video_path.stat().st_size < 1000:
        raise RuntimeError("The annotated video did not render correctly.")

    # Keep the compact sampled-frame previews in the temp folder for the in-app frame explorer.
    # They are not copied to Drive or included in the user-facing CSV.

    clean_rows = []
    for item in temporal_rows:
        clean = {k: v for k, v in item.items() if k not in {"heatmap", "rl_observation", "model_input_path"}}
        clean_rows.append(clean)
    frame_table = pd.DataFrame(clean_rows)
    frame_table = frame_table.drop(columns=[c for c in ["source_frame_path", "heat_overlay_path"] if c in frame_table.columns])
    if "is_anomaly" not in frame_table.columns:
        frame_table["is_anomaly"] = frame_table["anomaly_score"].astype(float) >= float(decision_boundary)
    else:
        frame_table["is_anomaly"] = frame_table["is_anomaly"].astype(bool)
    frame_table["detector_prediction"] = np.where(frame_table["is_anomaly"], "ANOMALY", "NORMAL")
    frame_table["decision_boundary"] = float(decision_boundary)

    csv_bytes = frame_table.to_csv(index=False).encode("utf-8")
    max_idx = int(frame_table["anomaly_score"].astype(float).idxmax())
    min_idx = int(frame_table["anomaly_score"].astype(float).idxmin())
    max_loc_row = frame_table.loc[max_idx]
    min_loc_row = frame_table.loc[min_idx]
    rl_enabled = inspection_mode == "RL-assisted inspection"
    action_counts = frame_table["ppo_final_action"].value_counts().to_dict() if rl_enabled and "ppo_final_action" in frame_table.columns else {}
    review_events = (
        int(frame_table["human_review_event"].sum())
        if rl_enabled and "human_review_event" in frame_table.columns
        else int(action_counts.get("HUMAN_REVIEW", 0)) if rl_enabled else 0
    )
    report = {
        "video_name": original_name,
        "category": active_category,
        "detector": CATEGORY_DETECTORS[active_category],
        "inspection_mode": inspection_mode,
        "roi_mode": roi_mode,
        "roi_custom_percentages": {
            "left": int(roi_x_pct), "top": int(roi_y_pct),
            "width": int(roi_w_pct), "height": int(roi_h_pct),
        } if roi_mode == "Custom ROI" else None,
        "source_fps": fps,
        "source_frame_count": frame_count,
        "source_duration_seconds": duration_sec,
        "sampled_frames": len(frame_table),
        "requested_sampling_interval_seconds": interval_sec,
        "effective_sampling_interval_seconds": effective_interval,
        "decision_boundary": float(decision_boundary),
        "decision_threshold_policy": getattr(active_engine, "decision_threshold_policy", "legacy fixed 0.95 fallback"),
        "decision_threshold_source": str(next((r.get("decision_threshold_source", "") for r in temporal_rows if r.get("decision_threshold_source")), "")),
        "high_anomaly_frame_count": int(frame_table["is_anomaly"].sum()),  # compatibility alias
        "predicted_anomaly_frame_count": int(frame_table["is_anomaly"].sum()),
        "frames_at_or_above_decision_threshold": int((frame_table.anomaly_score >= float(decision_boundary)).sum()),
        "frames_at_or_above_0_95": int((frame_table.anomaly_score >= 0.95).sum()),  # historical comparison, not the active cutoff
        "ppo_action_counts": action_counts,
        "human_review_event_count": review_events,
        "review_action_frames": int(action_counts.get("HUMAN_REVIEW", 0)),
        "flag_action_frames": int(action_counts.get("FLAG_ANOMALY", 0)),
        "monitor_action_frames": int(action_counts.get("MONITOR", action_counts.get("DO_NOTHING", 0))),
        "guardrail_reviews": int(frame_table["human_review_guardrail"].sum()) if "human_review_guardrail" in frame_table.columns else 0,
        "mean_anomaly_score": float(frame_table.anomaly_score.mean()),
        "median_anomaly_score": float(frame_table.anomaly_score.median()),
        "std_anomaly_score": float(frame_table.anomaly_score.std(ddof=0)),
        "score_saturated_at_one_count": int((frame_table["anomaly_score"].astype(float) >= 1.0 - 1e-9).sum()),
        "score_saturated_at_one_rate": float((frame_table["anomaly_score"].astype(float) >= 1.0 - 1e-9).mean()),
        "unique_calibrated_score_count": int(frame_table["anomaly_score"].astype(float).round(8).nunique()),
        "pre_final_score_min": float(pd.to_numeric(frame_table.get("pre_final_calibration_score", pd.Series(dtype=float)), errors="coerce").min()),
        "pre_final_score_max": float(pd.to_numeric(frame_table.get("pre_final_calibration_score", pd.Series(dtype=float)), errors="coerce").max()),
        "max_anomaly_score": float(frame_table.anomaly_score.max()),
        "min_anomaly_score": float(frame_table.anomaly_score.min()),
        "max_score_location": {"frame_index": int(max_loc_row["frame_index"]), "timestamp_sec": float(max_loc_row["timestamp_sec"])},
        "min_score_location": {"frame_index": int(min_loc_row["frame_index"]), "timestamp_sec": float(min_loc_row["timestamp_sec"])},
        "ppo_policy_mode": getattr(active_engine, "ppo_mode", "disabled") if rl_enabled else "disabled",
        "ppo_checkpoint": getattr(active_engine, "ppo_model_filename", "") if rl_enabled else "",
        "video_mime_type": video_mime_type,
        "browser_playback_ready": bool(playback_warning is None),
        "research_caveat": "Visual detector scores are normal-reference percentiles, not defect probabilities. PPO V4 and the Temporal Transformer were trained/evaluated on synthetic sequences derived from MVTec still images rather than continuous production video; RL actions are experimental triage outputs.",
    }
    dashboard_bytes = build_dashboard_html(report, frame_table)
    report_bytes = json.dumps(report, indent=2).encode("utf-8")
    results_zip = workdir / "sentinel_inspection_results.zip"
    with zipfile.ZipFile(results_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(processed_video_path, arcname=processed_video_path.name)
        zf.writestr("frame_level_report.csv", csv_bytes)
        zf.writestr("summary.json", report_bytes)
        zf.writestr("inspection_dashboard.html", dashboard_bytes)
    progress.progress(1.0, text="Inspection complete.")
    status.success("Inspection finished. Review the frame timeline and download your outputs below.")
    return {
        "workdir": workdir,
        "video_path": processed_video_path,
        "csv_bytes": csv_bytes,
        "report_bytes": report_bytes,
        "zip_path": results_zip,
        "report": report,
        "table": frame_table,
        "rows": temporal_rows,
        "meta": meta,
        "output_fps": output_fps,
        "video_mime_type": video_mime_type,
        "playback_warning": playback_warning,
        "dashboard_bytes": dashboard_bytes,
    }



# ------------------------- Primary Navigation Tabs -------------------------
tab_video, tab_rag, tab_copilot, tab_benchmark = st.tabs([
    "INSPECTION",
    "QUALITY AUDIT",
    "OPERATOR COPILOT",
    "MODEL BENCHMARK"
])

with tab_video:
    tab_image_mode, tab_stream_mode = st.tabs([
        "IMAGE INSPECTION // MVTec",
        "VIDEO FEED // TEMPORAL TRIAGE"
    ])

    with tab_image_mode:
        st.markdown("<div class='section-kicker'>Image input</div>", unsafe_allow_html=True)
        st.caption("Scores use live inference. Bundled examples are illustrative, not held-out evaluation records.")

        # Quick In-Tab Pipeline Switcher
        st.markdown("<div style='margin-bottom:8px;font-weight:600;color:#9cabb3;font-size:0.76rem;letter-spacing:.08em'>INSPECTION PIPELINE</div>", unsafe_allow_html=True)
        def _set_active_suite(suite: str) -> None:
            st.session_state["active_architecture_suite"] = suite
            st.session_state["sidebar_suite_select"] = suite

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.button(
                "Pipeline B · Deep visual", type="primary" if is_pipeline_b else "secondary",
                width="stretch", key="btn_sel_pipe_b", on_click=_set_active_suite,
                args=(PIPELINE_SUITES[0],),
            )
        with col_b2:
            st.button(
                "Pipeline A · Statistical", type="primary" if is_pipeline_a else "secondary",
                width="stretch", key="btn_sel_pipe_a", on_click=_set_active_suite,
                args=(PIPELINE_SUITES[1],),
            )
        with col_b3:
            st.button(
                "Compare pipelines", type="primary" if is_benchmark_mode else "secondary",
                width="stretch", key="btn_sel_bench", on_click=_set_active_suite,
                args=(PIPELINE_SUITES[2],),
            )

        active_is_pipe_a = is_pipeline_a
        active_is_pipe_b = is_pipeline_b
        active_is_bench = is_benchmark_mode

        col_img_left, col_img_right = st.columns([1.1, 0.9], gap="large")
        with col_img_left:
            img_source = st.radio(
                "Select image input:",
                ["Built-in MVTec Samples", "Upload Custom Image (.png, .jpg)"],
                horizontal=True,
                key="img_source_choice"
            )
            img_bgr = None
            img_name = ""
            if img_source == "Built-in MVTec Samples":
                sample_type = st.radio("Condition:", ["Normal (Good)", "Defective (Anomaly)"], horizontal=True, key="img_sample_type")
                status_str = "good" if "Good" in sample_type else "defective"
                sample_file = assets_path / "sample_images" / f"mvtec_{category}_{status_str}.png"
                if sample_file.is_file():
                    img_bgr = cv2.imread(str(sample_file))
                    img_name = sample_file.name
                    st.caption(f"Loaded bundled visual example `{sample_file.name}` ({img_bgr.shape[1]}x{img_bgr.shape[0]} px). This is not a held-out MVTec evaluation record; displayed scores come from the selected detector.")
            else:
                uploaded_file = st.file_uploader("Drop an MVTec image here", type=["png", "jpg", "jpeg"], key="img_uploader")
                if uploaded_file is not None:
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    img_name = uploaded_file.name
                    if img_bgr is None:
                        st.error("This file could not be decoded as a PNG or JPEG. Choose a valid image and try again.")

        with col_img_right:
            st.markdown("<div class='section-kicker'>Active route</div>", unsafe_allow_html=True)
            if active_is_pipe_a:
                st.markdown(
                    f"<div class='panel'><div style='font-size:1.05rem;font-weight:800;color:#f8fafc'>{category.title()} · <span style='color:#60a5fa'>Pipeline A</span></div>"
                    f"<div style='color:#fbbf24;margin-top:6px;font-weight:700'>Live model unavailable</div>"
                    f"<div class='muted' style='margin-top:8px'>The ResNet-18/Mahalanobis inference export and trained Q-table are not present. This route is disabled rather than showing synthetic scores.</div></div>",
                    unsafe_allow_html=True
                )
            elif active_is_pipe_b:
                st.markdown(
                    f"<div class='panel'><div style='font-size:1.05rem;font-weight:800;color:#f8fafc'>{category.title()} · <span style='color:#5eead4'>Pipeline B</span></div>"
                    f"<div style='color:#5eead4;margin-top:6px;font-weight:700'>{CATEGORY_DETECTORS[category]}</div>"
                    f"<div class='muted' style='margin-top:8px'>Subtype model: DINOv2 + LinearSVM</div>"
                    f"<div class='muted' style='margin-top:6px'>Decision cutoff: {decision_boundary:.4f} · image triage is score-based; video uses PPO only when a checkpoint is present.</div></div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='panel'><div style='font-size:1.05rem;font-weight:800;color:#f8fafc'>{category.title()} · <span style='color:#f59e0b'>Dual Benchmark</span></div>"
                    f"<div style='margin-top:6px;font-size:0.9rem'><b style='color:#60a5fa'>Pipeline A</b>: unavailable · trained inference artifacts missing</div>"
                    f"<div style='margin-top:3px;font-size:0.9rem'><b style='color:#5eead4'>Pipeline B</b>: {CATEGORY_DETECTORS[category]} · live image scoring</div>"
                    f"<div class='muted' style='margin-top:8px'>This run reports the available Pipeline B result and marks Pipeline A unavailable; it does not fabricate a comparison.</div></div>",
                    unsafe_allow_html=True
                )

        if img_bgr is not None:
            btn_label = "Run Dual-Architecture Benchmark" if is_benchmark_mode else ("Analyze Image (Pipeline A)" if is_pipeline_a else "Analyze Image (Pipeline B)")
            if active_is_pipe_a:
                st.warning("Pipeline A live inference is disabled because this checkout does not include its exported Mahalanobis estimator or trained Q-table.")
            analyze_clicked = st.button(
                btn_label, type="primary", width="stretch", key="btn_analyze_img",
                disabled=active_is_pipe_a,
                help="Pipeline A's trained inference artifacts are not included in this checkout." if active_is_pipe_a else None,
            )
            if analyze_clicked:
                with st.spinner("Scoring the selected image with the Pipeline B visual detector..."):
                    engine = load_engine_cached(
                        str(project_path.resolve()), device_option, enable_rl,
                        _threshold_file_signature(assets_path / "decision_thresholds.json"),
                    )

                    # Pipeline A's ResNet-18/Mahalanobis model and trained Q-table are
                    # not bundled in this checkout. Never synthesize a result for them.
                    pipeline_a_result = {
                        "available": False,
                        "reason": "Mahalanobis model export and trained Q-table are not bundled.",
                    }

                    score_res = engine.score_frame(img_bgr, category)
                    score_b = float(score_res["anomaly_score"])
                    is_anom_b = bool(score_res["is_anomaly"])
                    heat = score_res["heatmap"]
                    subtype_b = "general"
                    if enable_subtypes and is_anom_b:
                        try:
                            sub_res = engine.predict_subtype(img_bgr, category)
                            if sub_res:
                                subtype_b = sub_res.get("predicted_subtype", "general")
                        except Exception as subtype_exc:
                            st.warning(f"Optional defect subtype could not be loaded: {subtype_exc}")

                    if score_b < 0.40:
                        ppo_action = "MONITOR"
                        ppo_action_badge = "MONITOR / SKIP"
                        ppo_action_desc = "Low Risk · Normal unit, zero inspection cost ($0)"
                        ppo_cost = 0.0
                    elif score_b < decision_boundary:
                        ppo_action = "FLAG_ANOMALY"
                        ppo_action_badge = "FLAG ANOMALY"
                        ppo_action_desc = "Medium Risk · Flagged by model, automated verification ($1)"
                        ppo_cost = 1.0
                    else:
                        ppo_action = "HUMAN_REVIEW"
                        ppo_action_badge = "HUMAN EXPERT REVIEW"
                        ppo_action_desc = "High Risk · Routed to human review by the image score tier ($5); this single-frame route does not run PPO."
                        ppo_cost = 5.0

                    # Spatial Heatmap overlay
                    if is_anom_b:
                        norm_heat = ((np.clip(heat, 0, 1)) * 255).astype(np.uint8)
                        color_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(img_bgr, 0.55, color_heat, 0.45, 0)
                    else:
                        norm_heat = np.full((img_bgr.shape[0], img_bgr.shape[1]), 18, dtype=np.uint8)
                        color_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)
                        overlay = cv2.addWeighted(img_bgr, 0.85, color_heat, 0.15, 0)

                    # Determine active values to pass to RAG / Copilot telemetry
                    if active_is_pipe_a:
                        eff_score = None
                        eff_anom = False
                        eff_action = "UNAVAILABLE"
                        eff_subtype = "Unavailable"
                        eff_det = "Pipeline A unavailable: inference artifacts missing"
                    else:
                        eff_score = score_b
                        eff_anom = is_anom_b
                        eff_action = ppo_action
                        eff_subtype = subtype_b
                        eff_det = CATEGORY_DETECTORS.get(category, "Visual Anomaly Detector")

                    single_row = {
                        "frame_index": 1,
                        "timestamp_sec": 0.0,
                        "anomaly_score": eff_score,
                        "is_anomaly": eff_anom,
                        "detector": eff_det,
                        "ppo_final_action": eff_action,
                        "predicted_subtype": eff_subtype,
                        "category": category,
                        "pipeline": "Pipeline A" if active_is_pipe_a else "Pipeline B" if active_is_pipe_b else "Benchmark",
                    }
                    table_df = pd.DataFrame([single_row])
                    rpt = {
                        "video_name": img_name,
                        "category": category,
                        "inspection_mode": f"Native MVTec Image Inspection ({model_suite})",
                        "detector": eff_det,
                        "decision_boundary": float(decision_boundary),
                        "ppo_policy_mode": "disabled",
                        "triage_policy": "score_threshold_image_triage",
                        "sampled_frames": 1,
                        "high_anomaly_frame_count": 1 if eff_anom else 0,
                        "mean_anomaly_score": eff_score,
                        "max_anomaly_score": eff_score,
                        "min_anomaly_score": eff_score,
                        "max_score_location": {"frame_index": 1, "timestamp_sec": 0.0},
                        "min_score_location": {"frame_index": 1, "timestamp_sec": 0.0},
                        "human_review_event_count": 1 if eff_action == "HUMAN_REVIEW" else 0,
                        "ppo_action_counts": {eff_action: 1},
                    }
                    st.session_state["sentinel_result"] = {
                        "table": table_df,
                        "report": rpt,
                        "rows": [single_row],
                    }
                    st.session_state["sentinel_category"] = category
                    st.session_state["single_image_analysis"] = {
                        "img_bgr": img_bgr,
                        "overlay_bgr": overlay,
                        "pipeline_a": pipeline_a_result,
                        "pipeline_b": {
                            "name": "Pipeline B",
                            "detector": CATEGORY_DETECTORS.get(category, "Visual Anomaly Detector"),
                            "score": score_b,
                            "is_anom": is_anom_b,
                            "action": ppo_action,
                            "action_badge": ppo_action_badge,
                            "action_desc": ppo_action_desc,
                            "cost": ppo_cost,
                            "subtype": subtype_b,
                            "score_saturated_at_one": bool(score_res.get("score_saturated_at_one", False)),
                            "pre_final_calibration_score": float(score_res.get("pre_final_calibration_score", score_b)),
                            "decision_threshold_source": str(score_res.get("decision_threshold_source", "")),
                        },
                        "active_suite": model_suite,
                        "category": category,
                        "decision_source": "score_threshold_image_triage",
                    }

        img_res = st.session_state.get("single_image_analysis")
        if img_res is not None and (
            img_res.get("active_suite") != model_suite or img_res.get("category") != category
        ):
            img_res = None
        if img_res is not None and isinstance(img_res.get("pipeline_b"), dict):
            st.markdown("---")
            st.markdown("### Inspection & Scheduling Analysis")
            
            p_a = img_res["pipeline_a"]
            p_b = img_res["pipeline_b"]

            if active_is_bench:
                # ── BENCHMARK MODE: SIDE-BY-SIDE EVALUATION ──
                st.markdown("<div style='font-size:1.15rem;font-weight:800;color:#f59e0b;margin-bottom:12px'>Dual Architecture Comparison</div>", unsafe_allow_html=True)
                img_c1, img_c2 = st.columns([1, 1])
                with img_c1:
                    st.image(cv2.cvtColor(img_res["img_bgr"], cv2.COLOR_BGR2RGB), caption="Original MVTec Input Image", width="stretch")
                with img_c2:
                    st.image(cv2.cvtColor(img_res["overlay_bgr"], cv2.COLOR_BGR2RGB), caption="Spatial Heatmap Evidence (STFPM / PatchCore)", width="stretch")

                bench_c1, bench_c2 = st.columns(2, gap="large")
                with bench_c1:
                    st.markdown("<div class='panel' style='border-left:4px solid #60a5fa'><b style='color:#60a5fa;font-size:1.05rem'>Pipeline A</b><br><span class='muted'>Live result unavailable</span></div>", unsafe_allow_html=True)
                    st.info(p_a.get("reason", "The required trained inference artifacts are unavailable."))

                with bench_c2:
                    st.markdown("<div class='panel' style='border-left:4px solid #5eead4'><b style='color:#5eead4;font-size:1.05rem'>Pipeline B</b><br><span class='muted'>Live visual detector; single-image action is score-tier routing</span></div>", unsafe_allow_html=True)
                    st.markdown(f"**Model Verdict**: `{'ANOMALY' if p_b['is_anom'] else 'NORMAL'}`")
                    st.metric("Calibrated Anomaly Score", f"{p_b['score']:.4f}", delta=f"{p_b['score'] - decision_boundary:+.3f} vs cutoff")
                    st.markdown(f"**Defect Subtype**: `{p_b['subtype'].title()}`")
                    st.markdown(f"**Score-tier action**: `{p_b['action_badge']}`")
                    st.caption(p_b["action_desc"])
                    st.metric("Pipeline B Inspection Cost", f"${p_b['cost']:.2f}")

                st.info("Policy agreement and cost delta are unavailable because Pipeline A did not produce a live result. Use the held-out telemetry benchmark for the recorded Pipeline A/Pipeline B comparison.")

            elif active_is_pipe_a:
                # ── PIPELINE A VIEW ──
                st.markdown("<div style='font-size:1.05rem;font-weight:700;color:#dce6eb;margin-bottom:12px'>Pipeline A · Statistical anomaly detection</div>", unsafe_allow_html=True)
                res_col1, res_col2, res_col3 = st.columns([1, 1, 1.2])
                with res_col1:
                    st.image(cv2.cvtColor(img_res["img_bgr"], cv2.COLOR_BGR2RGB), caption="Original MVTec Input Image", width="stretch")
                with res_col2:
                    st.image(cv2.cvtColor(img_res["overlay_bgr"], cv2.COLOR_BGR2RGB), caption="ResNet-18 Embedding Activation", width="stretch")
                with res_col3:
                    st.markdown(f"**Model Verdict**: `{'ANOMALY' if p_a['is_anom'] else 'NORMAL'}`")
                    st.metric("Mahalanobis Anomaly Score", f"{p_a['score']:.4f}", delta=f"{p_a['score'] - decision_boundary:+.3f} vs cutoff")
                    st.markdown(f"**Defect Subtype**: `{p_a['subtype'].title()}`")
                    st.markdown(f"**Q-Learning Triage Action**: `{p_a['action_badge']}`")
                    st.caption(p_a["action_desc"])
                    st.metric("Simulated Inspection Cost", f"${p_a['cost']:.2f}")

            else:
                # ── PIPELINE B VIEW ──
                st.markdown("<div style='font-size:1.15rem;font-weight:800;color:#5eead4;margin-bottom:12px'>Pipeline B · Deep visual inspection</div>", unsafe_allow_html=True)
                res_col1, res_col2, res_col3 = st.columns([1, 1, 1.2])
                with res_col1:
                    st.image(cv2.cvtColor(img_res["img_bgr"], cv2.COLOR_BGR2RGB), caption="Original MVTec Input Image", width="stretch")
                with res_col2:
                    st.image(cv2.cvtColor(img_res["overlay_bgr"], cv2.COLOR_BGR2RGB), caption="Anomaly Localization Heatmap", width="stretch")
                with res_col3:
                    st.markdown(f"**Model Verdict**: `{'ANOMALY' if p_b['is_anom'] else 'NORMAL'}`")
                    st.metric("Calibrated Anomaly Score", f"{p_b['score']:.4f}", delta=f"{p_b['score'] - decision_boundary:+.3f} vs cutoff")
                    st.markdown(f"**Defect Subtype**: `{p_b['subtype'].title()}`")
                    st.markdown(f"**Score-tier action**: `{p_b['action_badge']}`")
                    st.caption(p_b["action_desc"])
                    st.metric("Simulated Inspection Cost", f"${p_b['cost']:.2f}")

                if p_b.get("score_saturated_at_one"):
                    st.warning(
                        "Score saturation: this raw detector score is at or above the largest saved normal-reference score, so calibration maps it to 1.000. "
                        "Treat this result as outside the current calibration range, not as a calibrated defect probability. The bundled visual examples are illustrative; validate with representative MVTec/custom nominal images before operational use."
                    )

            st.success("Result available in Quality Audit, Operator Copilot, and Model Benchmark.")
        else:
            st.markdown("---")
            if is_benchmark_mode:
                st.info("Run an analysis to see the available Pipeline B result. Pipeline A and live policy agreement remain unavailable until Pipeline A inference artifacts are exported.")
            elif is_pipeline_a:
                st.warning("Pipeline A live inference cannot run until its trained ResNet-18/Mahalanobis export and Q-table are added to this checkout.")
            else:
                st.info("**Pipeline B Ready**: Click **Analyze Image (Pipeline B)** above to run the visual detector, heatmap, and score-tier action. Single-image mode does not run PPO.")


    with tab_stream_mode:
        left, right = st.columns([1.15, 0.85], gap="large")
        with left:
            st.markdown("<div class='section-kicker'>Video input</div>", unsafe_allow_html=True)
            video_source_mode = st.radio(
                "Select inspection video source:",
                ["Built-in MVTec Dataset Sequence", "Upload Custom Video File"],
                horizontal=True,
                help="Select the built-in MVTec sequence for instant zero-setup inspection, or upload your own video file."
            )
    
            class DemoVideoPayload:
                def __init__(self, name: str, data: bytes):
                    self.name = name
                    self._data = data
                def getvalue(self) -> bytes:
                    return self._data
                def getbuffer(self) -> bytes:
                    return self._data
    
            uploaded_video = None
            if video_source_mode == "Built-in MVTec Dataset Sequence":
                demo_path = assets_path / "demo_videos" / f"mvtec_{category}_sample_conveyor.mp4"
                if demo_path.is_file():
                    uploaded_video = DemoVideoPayload(demo_path.name, demo_path.read_bytes())
                    st.caption(f"Loaded bundled conveyor preview for **{category.title()}** (40 frames @ 10 fps). This is an illustrative clip, not a held-out MVTec video benchmark.")
                else:
                    st.warning(f"Built-in sample video for {category} not found at {demo_path}.")
            else:
                uploaded_video = st.file_uploader(
                    "Drop an inspection video here",
                    type=["mp4", "mov", "avi", "mkv", "mpeg", "mpg"],
                    help="Short clips are best for the first run. Sampling is controlled in the sidebar.",
                    label_visibility="collapsed",
                )
        with right:
            st.markdown("<div class='section-kicker'>Active route</div>", unsafe_allow_html=True)
            if is_pipeline_a:
                st.markdown(
                    f"<div class='panel'><div style='font-size:1.05rem;font-weight:800;color:#f8fafc'>{category.title()} · <span style='color:#60a5fa'>Pipeline A</span></div>"
                    f"<div style='color:#fbbf24;margin-top:6px;font-weight:700'>Live model unavailable</div>"
                    f"<div class='muted' style='margin-top:9px'>The Pipeline A exports are not bundled. Video inference is available under Pipeline B only.</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                if not enable_rl:
                    image_action_label = "Frames → calibrated anomaly score → ROI-aligned heatmap"
                elif v4_policy.is_file():
                    image_action_label = "Video: learned PPO V4 action policy"
                elif legacy_policy.is_file():
                    image_action_label = "Video: legacy PPO action policy"
                else:
                    image_action_label = "Video: adaptive triage fallback (no PPO checkpoint)"
                st.markdown(
                    f"<div class='panel'><div style='font-size:1.05rem;font-weight:800;color:#f8fafc'>{category.title()} · <span style='color:#5eead4'>Pipeline B</span></div><div style='color:#5eead4;margin-top:6px;font-weight:700'>{CATEGORY_DETECTORS[category]}</div><div class='muted' style='margin-top:9px'>{image_action_label}</div></div>",
                    unsafe_allow_html=True,
                )
    
        if category == "grid":
            st.caption("This category is routed to the trained STFPM ResNet-18 student–teacher detector.")
        else:
            st.caption("This category is routed to your trained YOLO26 global embedding + PatchCore-style local patch detector.")
    
    
        if uploaded_video is not None:
            preview_file = None
            preview_cap = None
            try:
                preview_file = tempfile.NamedTemporaryFile(suffix=Path(uploaded_video.name).suffix or ".mp4", delete=False)
                preview_file.write(uploaded_video.getvalue())
                preview_file.close()
                browser_bytes, browser_mime, preview_warning = cached_browser_video(
                    uploaded_video.getvalue(), Path(uploaded_video.name).suffix.lower()
                )
                st.video(browser_bytes, format=browser_mime)
                st.caption("Source video preview. A browser-compatible copy is used when FFmpeg is available.")
                if preview_warning:
                    st.caption(preview_warning)
                preview_cap = cv2.VideoCapture(preview_file.name)
                ok_preview, first_preview = preview_cap.read()
                if ok_preview and first_preview is not None:
                    preview_bbox = compute_roi_bbox(
                        first_preview.shape[:2], roi_mode,
                        x_pct=roi_x_pct, y_pct=roi_y_pct,
                        w_pct=roi_w_pct, h_pct=roi_h_pct,
                    )
                    px1, py1, px2, py2 = preview_bbox
                    roi_preview = first_preview[py1:py2, px1:px2]
                    st.image(
                        cv2.cvtColor(resize_for_display(roi_preview, max_w=640, max_h=420), cv2.COLOR_BGR2RGB),
                        caption=f"Preview of model input region · {roi_mode} · source ROI {(px1, py1, px2, py2)}",
                        width="stretch",
                    )
                    if roi_mode == "Full frame" and abs(first_preview.shape[1] / max(first_preview.shape[0], 1) - 1.0) > 0.15:
                        st.warning(
                            "Full frame is not square, but the MVTec detectors were trained on square images and resize inputs to square. "
                            "This can distort the object and increase false positives; prefer Center square crop or Custom ROI."
                        )
            except Exception as preview_exc:
                st.caption(f"ROI preview unavailable: {preview_exc}")
            finally:
                if preview_cap is not None:
                    preview_cap.release()
                if preview_file is not None:
                    try:
                        os.unlink(preview_file.name)
                    except OSError:
                        pass
    
            video_mb = len(uploaded_video.getbuffer()) / (1024 ** 2)
            info_cols = st.columns(5)
            info_cols[0].metric("Video file", uploaded_video.name[:22])
            info_cols[1].metric("Upload size", f"{video_mb:.1f} MB")
            info_cols[2].metric("Mode", "RL-assisted" if enable_rl else "Detector only")
            info_cols[3].metric("Sampling", f"every {sampling_interval:g}s")
            info_cols[4].metric("Frame budget", str(max_frames))
            if is_pipeline_a or is_benchmark_mode:
                st.info("Video inspection runs Pipeline B only. Select Pipeline B explicitly; video mode cannot provide a Pipeline A or dual-pipeline live comparison.")
            run_clicked = st.button(
                "Inspect this video", type="primary", width="stretch",
                disabled=is_pipeline_a or is_benchmark_mode,
                help="Video inference uses Pipeline B only; choose Pipeline B to run it." if is_pipeline_a or is_benchmark_mode else None,
            )
            if run_clicked:
                with st.spinner("Loading visual detectors and running inspection route…"):
                    try:
                        engine = load_engine_cached(
                            str(project_path.resolve()), device_option, enable_rl,
                            _threshold_file_signature(assets_path / "decision_thresholds.json"),
                        )
                        result = process_video_file(
                            uploaded_video.getvalue(), uploaded_video.name, engine, category,
                            sampling_interval, max_frames, enable_subtypes,
                            roi_mode=roi_mode,
                            roi_x_pct=roi_x_pct, roi_y_pct=roi_y_pct,
                            roi_w_pct=roi_w_pct, roi_h_pct=roi_h_pct,
                            inspection_mode=inspection_mode,
                            decision_boundary=decision_boundary,
                        )
                        st.session_state["sentinel_result"] = result
                        st.session_state["sentinel_video_result"] = result
                        st.session_state["sentinel_category"] = category
                        st.session_state["sentinel_dashboard_requested"] = False
                    except Exception as exc:
                        st.error("Video inspection failed. The exception below identifies the failing component.")
                        st.exception(exc)
    
        # Results persist across Streamlit reruns caused by downloads and dashboard controls.
        result = st.session_state.get("sentinel_video_result")
        if result is not None and result.get("zip_path") is not None:
            st.markdown("---")
            st.markdown("<div class='section-kicker'>03 / Inspection results</div>", unsafe_allow_html=True)
            rpt = result["report"]
            table = result["table"]
            rl_mode_result = rpt.get("inspection_mode") == "RL-assisted inspection"
            actions = rpt.get("ppo_action_counts", {}) or {}
    
            st.markdown(
                f"<div class='panel'><b>{html_lib.escape(str(rpt.get('video_name', 'Inspection video')))}</b> · "
                f"{html_lib.escape(str(rpt.get('category', '')).title())} · "
                f"{html_lib.escape(str(rpt.get('inspection_mode', '')))}<div class='muted' style='margin-top:6px'>"
                f"Detector: {html_lib.escape(str(rpt.get('detector', '')))} · ROI: {html_lib.escape(str(rpt.get('roi_mode', '')))} · "
                f"Sampled cadence: {float(rpt.get('effective_sampling_interval_seconds', 0)):.3f}s</div></div>",
                unsafe_allow_html=True,
            )
            if result.get("playback_warning"):
                st.warning(result["playback_warning"])
    
            if rpt.get("high_anomaly_frame_count", 0) >= max(1, int(0.70 * rpt.get("sampled_frames", 0))):
                st.warning(
                    f"At least 70% of sampled frames were labelled ANOMALY under this category’s decision policy (score ≥ {decision_boundary:.3f}). "
                    "This can indicate unusual footage, but may also indicate domain shift, incorrect category, lighting differences, or ROI/background mismatch. "
                    "The score is a normal-reference percentile, not a probability of defect."
                )
    
            metric_cols = st.columns(6 if rl_mode_result else 5)
            metric_cols[0].metric("Frames analysed", f"{rpt['sampled_frames']:,}")
            metric_cols[1].metric("Predicted anomalies", f"{rpt['high_anomaly_frame_count']:,}", help=f"Category decision policy: calibrated anomaly score ≥ {decision_boundary:.3f}.")
            metric_cols[2].metric("Mean score", f"{rpt['mean_anomaly_score']:.3f}")
            metric_cols[3].metric("Peak score", f"{rpt['max_anomaly_score']:.3f}", help=f"Frame {rpt['max_score_location']['frame_index']} at {rpt['max_score_location']['timestamp_sec']:.2f}s")
            metric_cols[4].metric("Lowest score", f"{rpt['min_anomaly_score']:.3f}", help=f"Frame {rpt['min_score_location']['frame_index']} at {rpt['min_score_location']['timestamp_sec']:.2f}s")
            if rl_mode_result:
                metric_cols[5].metric("Human-review events", f"{rpt.get('human_review_event_count', 0):,}", help=f"{rpt.get('review_action_frames', 0)} sampled frame(s) were labelled HUMAN_REVIEW")
    
            # Score timeline: fixed [0, 1] axis prevents Streamlit auto-scaling from making
            # score/cutoff lines look coincident or misleading. The raw/pre-final score chart
            # below makes ECDF saturation visible without changing the calibrated scores.
            timeline = table.copy()
            timeline["timestamp_sec"] = pd.to_numeric(timeline["timestamp_sec"], errors="coerce")
            timeline["frame_index"] = pd.to_numeric(timeline["frame_index"], errors="coerce")
            timeline["anomaly_score"] = pd.to_numeric(timeline["anomaly_score"], errors="coerce")
            timeline = timeline.replace([np.inf, -np.inf], np.nan).dropna(subset=["timestamp_sec", "anomaly_score"])
            timeline = timeline.sort_values(["timestamp_sec", "frame_index"], kind="stable")
            st.markdown("### Calibrated anomaly-score timeline")
            st.caption("Fixed vertical axis: 0.00–1.00. The dashed amber line is the active category-specific decision cutoff; the score line is not a defect probability.")
    
            series_parts = [timeline[["frame_index", "timestamp_sec", "anomaly_score"]].rename(columns={"anomaly_score": "value"}).assign(series="Calibrated anomaly score")]
            series_names = ["Calibrated anomaly score"]
            series_colors = ["#2dd4bf"]
            if rl_mode_result and "temporal_anomaly_probability" in timeline.columns:
                temporal_values = pd.to_numeric(timeline["temporal_anomaly_probability"], errors="coerce")
                if temporal_values.notna().any():
                    series_parts.append(timeline[["frame_index", "timestamp_sec"]].assign(value=temporal_values).dropna(subset=["value"]).assign(series="Temporal Transformer context"))
                    series_names.append("Temporal Transformer context")
                    series_colors.append("#a78bfa")
            plot_data = pd.concat(series_parts, ignore_index=True)
            line_chart = alt.Chart(plot_data).mark_line(point=True, strokeWidth=2.5).encode(
                x=alt.X("timestamp_sec:Q", title="Video timestamp (seconds)", axis=alt.Axis(tickCount=8, format=".1f")),
                y=alt.Y("value:Q", title="Score (0–1)", scale=alt.Scale(domain=[0, 1], clamp=True, nice=False), axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1.0], format=".2f")),
                color=alt.Color("series:N", title="Series", scale=alt.Scale(domain=series_names, range=series_colors)),
                detail=alt.Detail("series:N"),
                order=alt.Order("timestamp_sec:Q"),
                tooltip=[alt.Tooltip("frame_index:Q", title="Source frame", format=".0f"), alt.Tooltip("timestamp_sec:Q", title="Time (s)", format=".3f"), alt.Tooltip("series:N", title="Series"), alt.Tooltip("value:Q", title="Score", format=".5f")],
            )
            threshold_rule = alt.Chart(pd.DataFrame({"threshold": [float(decision_boundary)]})).mark_rule(
                color="#f59e0b", strokeDash=[7, 4], strokeWidth=2.5
            ).encode(y=alt.Y("threshold:Q", scale=alt.Scale(domain=[0, 1], clamp=True, nice=False)))
            score_chart = alt.layer(line_chart, threshold_rule).resolve_scale(y="shared").properties(height=280).interactive()
            st.altair_chart(score_chart, width="stretch")
    
            score_values = timeline["anomaly_score"].astype(float).to_numpy()
            saturated_mask = score_values >= (1.0 - 1e-9)
            saturated_count = int(saturated_mask.sum())
            saturation_rate = float(saturated_mask.mean()) if len(saturated_mask) else 0.0
            unique_count = int(pd.Series(score_values).round(8).nunique()) if len(score_values) else 0
            diag1, diag2, diag3, diag4 = st.columns(4)
            diag1.metric("Minimum calibrated score", f"{float(np.min(score_values)):.4f}" if len(score_values) else "—")
            diag2.metric("Maximum calibrated score", f"{float(np.max(score_values)):.4f}" if len(score_values) else "—")
            diag3.metric("Unique score values", f"{unique_count:,}")
            diag4.metric("Scores at 1.000", f"{saturated_count:,} ({saturation_rate:.1%})")
            if saturated_count:
                st.info(
                    f"{saturated_count} of {len(score_values)} sampled frame(s) ({saturation_rate:.1%}) scored exactly 1.000. "
                    "ECDF maps scores at or above the maximum saved normal reference to 1.000. Treat this run as outside the available calibration range; do not interpret saturation alone as a confirmed defect. "
                    "The fixed 0–1 chart and the pre-final-calibration score chart are provided for diagnosis."
                )
            if "pre_final_calibration_score" in timeline.columns:
                raw_plot = timeline[["frame_index", "timestamp_sec", "pre_final_calibration_score"]].copy()
                raw_plot["pre_final_calibration_score"] = pd.to_numeric(raw_plot["pre_final_calibration_score"], errors="coerce")
                raw_plot = raw_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=["pre_final_calibration_score"])
                if not raw_plot.empty:
                    raw_values = raw_plot["pre_final_calibration_score"].to_numpy(dtype=float)
                    raw_min, raw_max = float(np.min(raw_values)), float(np.max(raw_values))
                    raw_span = raw_max - raw_min
                    raw_scale = max(abs(raw_min), abs(raw_max), raw_span, 1e-8)
                    raw_pad = max(raw_span * 0.10, raw_scale * 0.02, 1e-8)
                    st.markdown("### Score before final ECDF calibration")
                    st.caption("Diagnostic view with its own automatically padded y-axis. For grid this is the STFPM residual score; for the other categories it is the combined YOLO/PatchCore percentile score that enters the final ECDF. Compare it within the selected category, not across detector types.")
                    raw_chart = alt.Chart(raw_plot).mark_line(point=True, strokeWidth=2.2, color="#60a5fa").encode(
                        x=alt.X("timestamp_sec:Q", title="Video timestamp (seconds)", axis=alt.Axis(tickCount=8, format=".1f")),
                        y=alt.Y("pre_final_calibration_score:Q", title="Pre-final-calibration score", scale=alt.Scale(domain=[raw_min - raw_pad, raw_max + raw_pad], nice=False, zero=False)),
                        tooltip=[alt.Tooltip("frame_index:Q", title="Source frame", format=".0f"), alt.Tooltip("timestamp_sec:Q", title="Time (s)", format=".3f"), alt.Tooltip("pre_final_calibration_score:Q", title="Pre-final-calibration score", format=".8g")],
                    ).properties(height=220).interactive()
                    st.altair_chart(raw_chart, width="stretch")
            threshold_source = next((str(r.get("decision_threshold_source", "")) for r in result.get("rows", []) if r.get("decision_threshold_source")), "category thresholds")
            policy_mode = str(rpt.get("ppo_policy_mode", "disabled"))
            if policy_mode == "v4_recall_balanced":
                policy_caption = "Learned PPO V4 uses its own internal temporal score reference."
            elif policy_mode == "adaptive_triage_fallback":
                policy_caption = "Adaptive triage fallback is active because no PPO checkpoint is bundled; it is not a learned PPO policy."
            elif policy_mode == "legacy_transformer":
                policy_caption = "Legacy PPO is active with its saved observation contract."
            else:
                policy_caption = "RL decisions are disabled for this run."
            st.caption(f"Active category: {category}. Decision cutoff: {decision_boundary:.4f}. Threshold source: {threshold_source}. Calibrated score = empirical percentile against normal-reference images, not a defect probability. {policy_caption}")
    
            dashboard_col, hide_col, export_col = st.columns([1, 1, 2])
            with dashboard_col:
                if st.button("Create dashboard", type="primary", width="stretch", key="create_dashboard"):
                    st.session_state["sentinel_dashboard_requested"] = True
            with hide_col:
                if st.button("Hide dashboard", width="stretch", key="hide_dashboard"):
                    st.session_state["sentinel_dashboard_requested"] = False
            with export_col:
                zip_bytes = result["zip_path"].read_bytes() if isinstance(result.get("zip_path"), Path) and result["zip_path"].is_file() else b""
                st.download_button(
                    "Export complete results pack",
                    data=zip_bytes,
                    file_name="sentinel_inspection_results.zip",
                    mime="application/zip",
                    width="stretch",
                )
    
            if st.session_state.get("sentinel_dashboard_requested", False):
                st.markdown("## Inspection dashboard")
                st.caption("Generated from this run's sampled frames. Use frame number and timestamp to locate any event in the source video.")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Predicted-anomaly rate", f"{100 * rpt['high_anomaly_frame_count'] / max(rpt['sampled_frames'], 1):.1f}%")
                d2.metric("Score spread", f"{rpt['max_anomaly_score'] - rpt['min_anomaly_score']:.3f}")
                d3.metric("Score median", f"{rpt.get('median_anomaly_score', 0):.3f}")
                d4.metric("Score std. dev.", f"{rpt.get('std_anomaly_score', 0):.3f}")
                dashboard_left, dashboard_right = st.columns([1.45, 1], gap="large")
                with dashboard_left:
                    st.markdown("### Highest anomaly-score frames")
                    top_cols = [c for c in ["frame_index", "timestamp_sec", "sample_index", "anomaly_score", "detector_prediction", "is_anomaly", "ppo_final_action", "human_review_event"] if c in table.columns]
                    top_anomalies = table.sort_values("anomaly_score", ascending=False).head(12)
                    st.dataframe(top_anomalies[top_cols], width="stretch", hide_index=True)
                with dashboard_right:
                    st.markdown("### Score distribution")
                    score_hist = pd.cut(
                        table["anomaly_score"].astype(float),
                        bins=[-0.001, 0.50, 0.70, 0.85, 0.95, 1.001],
                        labels=["0–.50", ".50–.70", ".70–.85", ".85–.95", ".95–1.00"],
                        include_lowest=True,
                    ).value_counts(sort=False).rename_axis("score_band").to_frame("frames")
                    st.bar_chart(score_hist)
                if rl_mode_result:
                    action_left, action_right = st.columns([1, 1.4], gap="large")
                    with action_left:
                        st.markdown("### Action mix")
                        action_df = pd.Series(actions, name="sampled frames").rename_axis("action").to_frame()
                        if not action_df.empty:
                            st.bar_chart(action_df)
                        st.caption(f"Controller: {rpt.get('ppo_checkpoint') or rpt.get('ppo_policy_mode')}. Human-review events are deduplicated using the policy cooldown; action-frame counts may be higher.")
                    with action_right:
                        st.markdown("### Action locations")
                        action_cols = [c for c in ["frame_index", "timestamp_sec", "anomaly_score", "ppo_raw_action", "ppo_final_action", "human_review_event", "review_cooldown_active_before_action"] if c in table.columns]
                        action_rows = table[table["ppo_final_action"].astype(str).isin(["FLAG_ANOMALY", "HUMAN_REVIEW"])] if "ppo_final_action" in table.columns else table.head(0)
                        st.dataframe(action_rows[action_cols], width="stretch", hide_index=True)
                    review_rows = table[table["ppo_final_action"].astype(str).eq("HUMAN_REVIEW")] if "ppo_final_action" in table.columns else table.head(0)
                    st.markdown("### Human-review queue")
                    review_cols = [c for c in ["frame_index", "timestamp_sec", "anomaly_score", "score_delta", "human_review_event", "previous_3s_mean_score", "persistent_high_5s_signal", "sharp_rise_signal"] if c in review_rows.columns]
                    if len(review_rows):
                        st.dataframe(review_rows[review_cols], width="stretch", hide_index=True)
                    else:
                        st.info("This replay contains no HUMAN_REVIEW action. The app does not fabricate a review event.")
                st.download_button(
                    "Download standalone HTML dashboard",
                    data=result.get("dashboard_bytes", b"<html><body>Dashboard unavailable</body></html>"),
                    file_name="sentinel_inspection_dashboard.html",
                    mime="text/html",
                    width="stretch",
                )
    
            tab_names = ["01 // ANNOTATED VIDEO", "02 // FRAME EXPLORER", "FRAME REPORT"]
            if rl_mode_result:
                tab_names.append("03 // RL DECISIONS")
            tab_names.extend([
                "RAG QUALITY AUDIT",
                "OPERATOR COPILOT",
                "MODEL BENCHMARK",
                "INTERPRETATION",
            ])
            tabs = st.tabs(tab_names)
            with tabs[0]:
                st.video(result["video_path"].read_bytes(), format=result.get("video_mime_type", "video/mp4"))
                st.download_button(
                    "Download annotated video",
                    data=result["video_path"].read_bytes(),
                    file_name=result["video_path"].name,
                    mime=result.get("video_mime_type", "video/mp4"),
                    width="stretch",
                )
                st.caption("Annotated inspection summary video: the original sequence is sampled at the selected cadence, not played frame-for-frame at native FPS.")
            with tabs[1]:
                frame_i = st.slider("Choose analysed frame", 0, max(0, len(table) - 1), 0, key="frame_explorer_idx")
                row = result["rows"][frame_i]
                frame_path = Path(row.get("source_frame_path", ""))
                heat_path = Path(row.get("heat_overlay_path", ""))
                if frame_path.exists() and heat_path.exists():
                    a, b = st.columns(2)
                    a.image(cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB), caption=f"Source frame · frame {row.get('frame_index')} · t={float(row.get('timestamp_sec', 0)):.2f}s", width="stretch")
                    b.image(cv2.cvtColor(cv2.imread(str(heat_path)), cv2.COLOR_BGR2RGB), caption="Anomaly heatmap overlay · existing ROI mapping preserved", width="stretch")
                else:
                    st.info("Frame previews are not available in this work folder. Use the annotated video or download the frame report.")
                visible = table.iloc[frame_i]
                st.json({k: (None if pd.isna(v) else v) for k, v in visible.to_dict().items() if k not in ["stfpm_or_patch_details"]})
            with tabs[2]:
                st.dataframe(table, width="stretch", hide_index=True)
                st.download_button(
                    "Download frame-level CSV",
                    data=result["csv_bytes"],
                    file_name="sentinel_frame_level_report.csv",
                    mime="text/csv",
                    width="stretch",
                )
            next_tab_index = 3
            if rl_mode_result:
                with tabs[next_tab_index]:
                    st.markdown("### PPO action trace")
                    decision_cols = [c for c in [
                        "timestamp_sec", "frame_index", "anomaly_score", "score_delta", "ppo_raw_action", "ppo_final_action",
                        "human_review_event", "review_cooldown_active_before_action", "sharp_rise_signal",
                        "persistent_high_5s_signal", "escalation_candidate_signal", "ppo_decision_source",
                    ] if c in table.columns]
                    st.dataframe(table[decision_cols], width="stretch", hide_index=True)
                    policy_mode = str(rpt.get("ppo_policy_mode", "disabled"))
                    if policy_mode == "v4_recall_balanced":
                        st.caption("Actions come from the trained 20-feature PPO policy without a rule-based inference override. Review-event count is cooldown-aware.")
                    elif policy_mode == "adaptive_triage_fallback":
                        st.caption("Actions come from the explicit adaptive triage fallback because no PPO checkpoint is bundled. These are not learned PPO outputs.")
                    else:
                        st.caption("Actions come from the loaded legacy PPO checkpoint.")
                next_tab_index += 1
    
            # --- Integrated RAG & Benchmarking Tabs ---
            with tabs[next_tab_index]:
                render_rag_audit_tab(result, category, key_prefix="video")
            next_tab_index += 1
    
            with tabs[next_tab_index]:
                render_copilot_tab(result, category, key_prefix="video")
            next_tab_index += 1
    
            with tabs[next_tab_index]:
                render_benchmark_tab(result, category, key_prefix="video")
            next_tab_index += 1
    
            with tabs[next_tab_index]:
                st.markdown(
                    """
                    **How to read the output**
    
                    - **Anomaly score:** empirical percentile relative to saved normal calibration images. Higher means more unusual; it is not a defect probability.
                    - **Heatmap:** relative local model evidence. Color is normalized per frame for display; compare numeric scores, not colors across different frames.
                    - **ROI:** the existing ROI pipeline scores only the selected region and maps the heatmap back into that region. This version preserves that behavior.
                    - **Detection-only mode:** runs the visual detector and heatmap without loading or executing Transformer/PPO components.
                    - **RL-assisted mode:** uses the learned PPO policy only when its checkpoint is present; otherwise an explicitly labelled adaptive triage fallback is used. The fallback is not a learned policy.
                    - **Decision thresholds:** the five category-specific thresholds selected by F2 on `controller_dev` are bundled with the app. A project-local `assets/decision_thresholds.json` overrides them when present. The continuous score is a calibrated normal-reference percentile, not a defect probability.
                    - **Score saturation:** the empirical CDF maps values at or above the maximum normal-reference score to 1.000. The app reports how many frames saturate and plots the pre-final-calibration score separately for diagnosis.
    
                    **Research limitation:** the Temporal Transformer and PPO were trained on synthetic sequences derived from static MVTec AD still images, not continuous production footage. RL decisions on arbitrary videos remain experimental; confirm important findings with a human inspector.
                    """
                )
        else:
            st.markdown("---")
            st.markdown("<div class='section-kicker'>How it works</div>", unsafe_allow_html=True)
            step1, step2, step3, step4 = st.columns(4)
            step1.markdown("<div class='panel'><b>01 · Detect</b><br><span class='muted'>STFPM for grid; YOLO26 + PatchCore-style local memory for zipper, hazelnut, leather and toothbrush.</span></div>", unsafe_allow_html=True)
            step2.markdown("<div class='panel'><b>02 · Localize</b><br><span class='muted'>Calibrated anomaly score and ROI-aligned visual heatmap.</span></div>", unsafe_allow_html=True)
            step3.markdown("<div class='panel'><b>03 · Compare</b><br><span class='muted'>Track the model prediction, continuous score, maximum/minimum frame and timestamp.</span></div>", unsafe_allow_html=True)
            step4.markdown("<div class='panel'><b>04 · Triage (optional)</b><br><span class='muted'>Enable RL-assisted mode for PPO decisions when a checkpoint is present; otherwise the app labels and uses its adaptive fallback.</span></div>", unsafe_allow_html=True)
    
        st.markdown("<hr><div style='display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#73849b;font-size:.78rem'><span>SENTINEL · HYBRID VISUAL INSPECTION</span><span>Local inference · ROI-aligned heatmaps · Frame-traceable review</span></div>", unsafe_allow_html=True)
    

with tab_rag:
    latest_res = st.session_state.get("sentinel_result") or {}
    render_rag_audit_tab(latest_res, category, key_prefix="workspace")

with tab_copilot:
    latest_res = st.session_state.get("sentinel_result") or {}
    render_copilot_tab(latest_res, category, key_prefix="workspace")

with tab_benchmark:
    latest_res = st.session_state.get("sentinel_result") or {}
    render_benchmark_tab(latest_res, category, key_prefix="workspace")
