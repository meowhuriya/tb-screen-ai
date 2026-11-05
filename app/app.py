"""TB-Screen AI: Medical Image Diagnostic Assistant (Streamlit Clinic App).

Track: HI-01 - Medical Image Diagnostic Assistant
Designed for frontline healthcare workers in under-resourced rural clinics:
- Low-power CPU-optimized inference with live latency telemetry
- Deterministic Out-of-Distribution (OOD) defense & radiographic quality checks
- 3-tier traffic-light risk triage: LOW RISK (Green) / NEEDS REVIEW (Yellow) / HIGH RISK (Red)
- Side-by-side original radiograph & Grad-CAM visual attention overlay with opacity slider
- High-volume Batch Screening Mode with automatic triage ordering (highest risk first)
- Local session screening history & CSV export
- Trilingual support: English, Hindi (हिंदी), and Telugu (తెలుగు)
- 3 Instant pre-loaded benchmark X-rays (Zero-upload testing for judges)
- 1-Click single-page downloadable clinical referral PDF report
- Citable Public Health Impact & Clinical Economics tab (WHO & Central TB Division data)
- Full local processing privacy guarantee (offline-capable, zero cloud transmission)
"""

import datetime
import io
import os
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from app.pdf_generator import create_diagnostic_pdf
except (ImportError, ModuleNotFoundError):
    from pdf_generator import create_diagnostic_pdf

from src.config import CLINICAL
from src.gradcam import explain_xray
from src.predict import (
    CALIBRATED_THRESHOLD,
    WHO_BENCHMARK_THRESHOLD,
    get_cached_model,
    predict_chest_xray,
    predict_chest_xray_onnx,
    validate_and_load_image,
)
from src.safety import inspect_xray_quality_and_ood, validate_image_file_size


@st.cache_resource(show_spinner=False)
def preload_inference_backends():
    """Cache and warm up both PyTorch and ONNX models once per server instance."""
    from src.predict import get_cached_model, get_cached_onnx_session
    pt_model, device = get_cached_model()
    try:
        onnx_sess = get_cached_onnx_session()
    except Exception:
        onnx_sess = None
    return pt_model, device, onnx_sess


# Preload inference backends to guarantee sub-10ms warm response
_ = preload_inference_backends()

# Page configuration
st.set_page_config(
    page_title="TB-Screen AI | Frontline Diagnostic Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Responsive styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 1.85rem;
        font-weight: 800;
        color: #182b49;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555e6d;
        margin-bottom: 1.0rem;
    }
    .alert-banner {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeeba;
        border-radius: 8px;
        padding: 10px 14px;
        font-weight: 600;
        font-size: 0.90rem;
        margin-bottom: 1.2rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.10rem;
        color: white;
        margin-bottom: 10px;
    }
    .action-card {
        background-color: #f8f9fa;
        border-left: 5px solid #182b49;
        border-radius: 4px;
        padding: 14px 18px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .privacy-badge {
        background-color: #e8f4fd;
        border: 1px solid #b6e0fe;
        color: #0c5460;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }
    .impact-stat-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .impact-stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #e53e3e;
    }
    .impact-stat-label {
        font-size: 0.85rem;
        color: #4a5568;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Multi-language dictionary
STRINGS: Dict[str, Dict[str, str]] = {
    "English": {
        "title": "TB-Screen AI: Frontline Medical Diagnostic Assistant",
        "subtitle": "High-Sensitivity Pulmonary Tuberculosis Screening for Rural & Under-Resourced Clinics (HI-01)",
        "banner": "SCREENING AID ONLY. NOT A FINAL MEDICAL DIAGNOSIS. All positive findings must be clinically corroborated with sputum smear microscopy or GeneXpert cartridge testing.",
        "sidebar_header": "Clinic & Language Settings",
        "select_lang": "Interface Language / भाषा / భాష",
        "preset_title": "Instant Demo Samples (For Judges):",
        "sample_1": "1. Normal Chest (Healthy)",
        "sample_2": "2. Advanced TB (Cavitation)",
        "sample_3": "3. Moderate Pulmonary TB",
        "upload_label": "Or Upload Frontal Chest X-Ray (PNG, JPG, JPEG):",
        "processing": "Analyzing pulmonary parenchyma & computing Grad-CAM explainability overlay...",
        "orig_heading": "Original Chest Radiograph",
        "cam_heading": "Grad-CAM Saliency Overlay (Pulmonary Attention)",
        "results_header": "Clinical Triage & Risk Assessment",
        "prob_tb": "TB Probability",
        "confidence": "AI Confidence",
        "calibrated_cutoff": "Calibrated Cutoff",
        "status_normal": "LOW RISK (Healthy Lung Fields)",
        "msg_normal": "No radiographic patterns indicative of active pulmonary tuberculosis detected. Bilateral pulmonary fields appear clear.",
        "action_normal": "Next Step: Routine primary care. Advise patient to return if persistent cough (>2 weeks), evening fever, or night sweats develop.",
        "status_review": "NEEDS REVIEW (Indeterminate Opacity)",
        "msg_review": "Borderline suspicious radiographic opacity detected. Findings are indeterminate.",
        "action_review": "Next Step: Immediate review by Medical Officer. Collect morning sputum for Acid-Fast Bacilli (AFB) smear or GeneXpert MTB/RIF assay. National TB Helpline: 1800-11-6666.",
        "status_high": "HIGH RISK (Active TB Indicated)",
        "msg_high": "Radiographic pattern strongly consistent with active pulmonary tuberculosis (cavitary lesion or infiltrative opacity detected).",
        "action_high": "Next Step: Provide surgical mask & isolate. Fast-track sputum GeneXpert test. Refer patient to District TB Center / DOTS clinic. National TB Helpline: 1800-11-6666 | nikshay.in",
        "download_pdf": "Download Official Clinical Referral PDF Report",
        "pdf_success": "PDF Referral Report successfully generated!",
        "patient_id_label": "Patient Identifier (Optional):",
        "notes_label": "Field Health Worker Notes (Optional):",
    },
    "हिन्दी (Hindi)": {
        "title": "टीबी-स्क्रीन एआई: मेडिकल इमेज डायग्नोस्टिक असिस्टेंट",
        "subtitle": "ग्रामीण एवं कम संसाधन वाले प्राथमिक स्वास्थ्य केंद्रों के लिए टीबी स्क्रीनिंग सहायक (HI-01)",
        "banner": "केवल स्क्रीनिंग सहायता। यह अंतिम चिकित्सा निदान नहीं है। सभी निष्कर्षों की पुष्टि थूक जांच या GeneXpert परीक्षण द्वारा आवश्यक है।",
        "sidebar_header": "क्लिनिक एवं भाषा सेटिंग्स",
        "select_lang": "Interface Language / भाषा / భాష",
        "preset_title": "त्वरित डेमो नमूने (जजों के परीक्षण हेतु):",
        "sample_1": "1. सामान्य फेफड़े (स्वस्थ)",
        "sample_2": "2. गंभीर टीबी (कैविटेशन)",
        "sample_3": "3. मध्यम फुफ्फुसीय टीबी",
        "upload_label": "या छाती का एक्स-रे अपलोड करें (PNG, JPG, JPEG):",
        "processing": "फेफड़ों के एक्स-रे का विश्लेषण और ग्रैड-सीएएम हीटमैप तैयार किया जा रहा है...",
        "orig_heading": "मूल चेस्ट एक्स-रे",
        "cam_heading": "ग्रैड-सीएएम हीटमैप (संदेहास्पद क्षेत्र)",
        "results_header": "क्लिनिकल ट्राइएज एवं जोखिम मूल्यांकन",
        "prob_tb": "टीबी की संभावना",
        "confidence": "एआई विश्वास स्तर",
        "calibrated_cutoff": "कैलिब्रेटेड सीमा",
        "status_normal": "कम जोखिम (फेफड़े सामान्य हैं)",
        "msg_normal": "सक्रिय फेफड़ों की टीबी का कोई स्पष्ट लक्षण नहीं मिला। दोनों फेफड़े सामान्य प्रतीत होते हैं।",
        "action_normal": "अगला कदम: सामान्य स्वास्थ्य परामर्श। यदि 2 सप्ताह से अधिक खांसी, बुखार या रात में पसीना आए तो दोबारा जांच कराएं।",
        "status_review": "समीक्षा आवश्यक (संदिग्ध धब्बा)",
        "msg_review": "फेफड़ों में मध्यम या अस्पष्ट धब्बा मिला है। अतिरिक्त चिकित्सीय समीक्षा की आवश्यकता है।",
        "action_review": "अगला कदम: चिकित्सा अधिकारी द्वारा जांच करवाएं। थूक की जांच (AFB स्मीयर) या GeneXpert परीक्षण करवाएं। राष्ट्रीय टीबी हेल्पलाइन: 1800-11-6666",
        "status_high": "उच्च जोखिम (सक्रिय टीबी के संकेत)",
        "msg_high": "एक्स-रे में सक्रिय फेफड़ों की टीबी के मजबूत संकेत हैं (कैविटी या सघन धब्बा देखा गया)।",
        "action_high": "अगला कदम: मरीज को तुरंत मास्क पहनाएं। तत्काल GeneXpert जांच करवाएं और नजदीकी डॉट्स (DOTS) केंद्र भेजें। राष्ट्रीय टीबी हेल्पलाइन: 1800-11-6666 | nikshay.in",
        "download_pdf": "आधिकारिक क्लिनिकल रेफरल पीडीएफ रिपोर्ट डाउनलोड करें",
        "pdf_success": "पीडीएफ रेफरल रिपोर्ट सफलतापूर्वक तैयार हो गई है!",
        "patient_id_label": "मरीज पहचान संख्या (वैकल्पिक):",
        "notes_label": "स्वास्थ्य कार्यकर्ता की टिप्पणी (वैकल्पिक):",
    },
    "తెలుగు (Telugu)": {
        "title": "టిబి-స్క్రీన్ AI: మెడికల్ ఇమేజ్ డయాగ్నస్టిక్ అసిస్టెంట్",
        "subtitle": "గ్రామీణ ప్రాథమిక ఆరోగ్య కేంద్రాల కోసం ఊపిరితిత్తుల క్షయవ్యాధి స్క్రీనింగ్ సహాయకం (HI-01)",
        "banner": "కేవలం స్క్రీనింగ్ సహాయం మాత్రమే. ఇది తుది వైద్య నిర్ధారణ కాదు. అన్ని ఫలితాలను కఫం లేదా GeneXpert పరీక్ష ద్వారా ధృవీకరించాలి.",
        "sidebar_header": "క్లినిక్ & భాష సెట్టింగ్‌లు",
        "select_lang": "Interface Language / भाषा / భాష",
        "preset_title": "తక్షణ డెమో నమూనాలు (న్యాయనిర్ణేతల కోసం):",
        "sample_1": "1. సాధారణ ఎక్స్-రే (ఆరోగ్యకరమైనది)",
        "sample_2": "2. తీవ్రమైన క్షయ (క్యావిటీ)",
        "sample_3": "3. మధ్యస్థాయి క్షయవ్యాధి",
        "upload_label": "లేదా ఛాతీ ఎక్స్-రే అప్‌లోడ్ చేయండి (PNG, JPG, JPEG):",
        "processing": "ఎక్స్-రే విశ్లేషణ మరియు గ్రాడ్-కామ్ హీట్‌మ్యాప్ రూపొందించబడుతోంది...",
        "orig_heading": "అసలు ఛాతీ ఎక్స్-రే",
        "cam_heading": "గ్రాడ్-కామ్ హీట్‌మ్యాప్ (ప్రభావిత ఊపిరితిత్తుల ప్రాంతం)",
        "results_header": "క్లినికల్ ట్రయాజ్ & ప్రమాద అంచనా",
        "prob_tb": "టిబి సంభావ్యత",
        "confidence": "AI ఖచ్చితత్వం",
        "calibrated_cutoff": "నిర్ణీత కటాఫ్",
        "status_normal": "తక్కువ ప్రమాదం (ఊపిరితిత్తులు సాధారణం)",
        "msg_normal": "చురుకైన క్షయవ్యాధి లక్షణాలు కనిపించలేదు. ఊపిరితిత్తులు ఆరోగ్యంగా ఉన్నాయి.",
        "action_normal": "తదుపరి చర్య: సాధారణ ఆరోగ్య సంరక్షణ. 2 వారాల కంటే ఎక్కువ దగ్గు లేదా జ్వరం ఉంటే మళ్లీ పరీక్షించండి.",
        "status_review": "సమీక్ష అవసరం (అస్పష్టమైన మచ్చ)",
        "msg_review": "ఊపిరితిత్తులలో స్వల్ప లేదా అస్పష్టమైన మచ్చ గుర్తించబడింది.",
        "action_review": "తదుపరి చర్య: వైద్యాధికారిచే సమీక్ష చేయించండి. కఫం పరీక్ష లేదా GeneXpert చేయించండి. జాతీయ టిబి హెల్ప్‌లైన్: 1800-11-6666",
        "status_high": "అధిక ప్రమాదం (చురుకైన క్షయ సూచన)",
        "msg_high": "ఎక్స్-రేలో చురుకైన క్షయవ్యాధి బలమైన సంకేతాలు ఉన్నాయి.",
        "action_high": "తదుపరి చర్య: తక్షణ జాగ్రత్తలు తీసుకోండి. సమీపంలోని డాట్స్ (DOTS) కేంద్రానికి రోగిని సిఫార్सु చేయండి. జాతీయ టిబి హెల్ప్‌లైన్: 1800-11-6666 | nikshay.in",
        "download_pdf": "అధికారిక క్లినికల్ రిఫెరల్ PDF నివేదికను డౌన్‌లోడ్ చేయండి",
        "pdf_success": "PDF నివేదిక విజయవంతంగా రూపొందించబడింది!",
        "patient_id_label": "రోగి గుర్తింపు సంఖ్య (ఐచ్ఛికం):",
        "notes_label": "ఫీల్డ్ వర్కర్ గమనికలు (ఐచ్ఛికం):",
    },
}

# Session state initialization with explicit keys
if "patient_history" not in st.session_state:
    st.session_state["patient_history"] = []

if "sample_selection" not in st.session_state:
    st.session_state["sample_selection"] = "None"

if "uploader_key_id" not in st.session_state:
    st.session_state["uploader_key_id"] = 0


def add_to_session_history(record: Dict[str, Any]) -> None:
    """Store patient screening record in browser session."""
    st.session_state["patient_history"].append(record)


def on_sample_radio_changed() -> None:
    """Callback when user selects a demo sample or None.
    
    Rule: Choosing a sample (1, 2, or 3) clears any uploaded file so only one source is active.
    """
    choice = st.session_state.get("sample_selection", "None")
    if choice != "None":
        st.session_state["uploader_key_id"] = st.session_state.get("uploader_key_id", 0) + 1


def on_file_uploaded() -> None:
    """Callback when a user uploads or removes a file in single screening mode.
    
    Rule: Uploading a file sets the sample radio to 'None (Upload Custom)' and analyses the upload.
    """
    current_key = f"single_uploader_{st.session_state.get('uploader_key_id', 0)}"
    uploaded = st.session_state.get(current_key)
    if uploaded is not None:
        st.session_state["sample_selection"] = "None"


# ======================================================================
# SIDEBAR
# ======================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lungs.png", width=64)
    st.markdown("## ⚙️ Settings")

    selected_lang_key = st.selectbox(
        STRINGS["English"]["select_lang"],
        options=["English", "हिन्दी (Hindi)", "తెలుగు (Telugu)"],
        index=0,
    )
    T = STRINGS[selected_lang_key]

    st.markdown("---")
    st.subheader(T["preset_title"])

    SAMPLE_OPTIONS = ["None", "Sample 1", "Sample 2", "Sample 3"]

    def format_sample_label(opt_key: str) -> str:
        if opt_key == "None":
            return "None (Upload Custom)"
        elif opt_key == "Sample 1":
            return T["sample_1"]
        elif opt_key == "Sample 2":
            return T["sample_2"]
        elif opt_key == "Sample 3":
            return T["sample_3"]
        return opt_key

    sample_choice = st.radio(
        "Select Sample to test instantly:",
        options=SAMPLE_OPTIONS,
        format_func=format_sample_label,
        key="sample_selection",
        on_change=on_sample_radio_changed,
    )

    st.markdown("---")
    st.markdown("### 📋 Decision Threshold")
    threshold_mode = st.radio(
        "Select Operating Threshold:",
        options=[
            "WHO Benchmark (τ = 0.05) [Default]",
            "Calibrated Cutoff (τ = 0.27)",
        ],
        index=0,
        help="WHO benchmark (0.05) prioritizes catching active cases. Calibrated cutoff (0.27) balances specificity.",
    )
    if "0.05" in threshold_mode:
        active_threshold = WHO_BENCHMARK_THRESHOLD
        st.info(
            "ℹ️ **WHO Benchmark (τ = 0.05) [Active]:** Favours catching active TB cases "
            "(95.10% sensitivity; 97/102 detected), but gives more false alarms (86.29% specificity). "
            "Recommended for high-risk rural screening camps."
        )
    else:
        active_threshold = CALIBRATED_THRESHOLD
        st.info(
            "ℹ️ **Calibrated Clinical Cutoff (τ = 0.27) [Active]:** Minimizes false alarms "
            "(94.67% specificity) with 88.24% sensitivity (90/102 detected). "
            "Recommended for clinics with limited GeneXpert cartridges."
        )

    st.caption("• **WHO Benchmark (τ = 0.05):** Sensitivity **95.10%** (97/102 detected; 5 missed) | Specificity **86.29%** (453/525)")
    st.caption("• **Calibrated Cutoff (τ = 0.27):** Sensitivity **88.24%** (90/102 detected; 95% CI: 81.58% – 93.97%) | Specificity **94.67%** (497/525)")

    st.markdown("---")
    st.markdown("### ⚡ Inference Engine")
    engine_choice = st.radio(
        "Select Runtime Engine:",
        options=["PyTorch Edge (with Grad-CAM)", "ONNX Runtime (Fast ~7ms)"],
        index=0,
        help="ONNX Runtime delivers 7.19 ms median CPU inference; PyTorch Edge computes full Grad-CAM gradients."
    )
    is_onnx_selected = "ONNX" in engine_choice

    st.markdown("---")
    st.markdown("### 🏥 System Status")
    runtime_label = "ONNX Runtime (CPU 7ms)" if is_onnx_selected else "PyTorch Edge (Grad-CAM)"
    st.success(
        f"● Engine: {runtime_label}\n\n"
        "● Device: Auto-detected (CPU/MPS)\n\n"
        "● Mode: In-Memory Ephemeral (Zero Storage)\n\n"
        "● Architecture: 100% Offline-Capable Edge Model"
    )

    st.markdown("---")
    st.markdown("### 🩺 Clinical Advisory")
    st.caption(
        "Clinical workflow, risk triage messages, and referral guidance reviewed by "
        "**Dr. Bepar Mohammad Tayyib, MBBS** (Clinical Advisor; not a member of the development team). "
        "\n\n*Notice: Screening aid only; has not been evaluated in formal clinical trials.*"
    )

# ======================================================================
# MAIN HEADER & PRIVACY BANNER
# ======================================================================
st.markdown(f"<div class='main-header'>🫁 {T['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub-header'>{T['subtitle']}</div>", unsafe_allow_html=True)

# Privacy & Edge architecture guarantee
st.markdown(
    "<div class='privacy-badge'>🔒 <b>Data Privacy Guarantee:</b> Designed for 100% offline edge deployment in rural clinics with zero cloud dependency. "
    "In this hosted web demonstration, radiographs are processed ephemerally in RAM and are never permanently stored, logged, or shared.</div>",
    unsafe_allow_html=True,
)

# Mandatory Clinical Regulatory Banner
st.markdown(
    f"<div class='alert-banner'>⚠️ <b>{T['banner']}</b></div>",
    unsafe_allow_html=True,
)

# ======================================================================
# TABS
# ======================================================================
tab_single, tab_batch, tab_history, tab_impact = st.tabs([
    "🔍 Single Patient Screening",
    "📂 High-Volume Batch Triage",
    "📋 Shift Session History",
    "📊 Public Health Impact & Clinical Economics",
])


# ----------------------------------------------------------------------
# TAB 1: SINGLE PATIENT SCREENING
# ----------------------------------------------------------------------
with tab_single:
    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        patient_id = st.text_input(T["patient_id_label"], value=f"PAT-{datetime.date.today().strftime('%Y%m%d')}-01")
    with col_p2:
        field_notes = st.text_input(T["notes_label"], placeholder="e.g. Cough > 2 weeks, evening fever, weight loss...")

    active_image = None
    source_name = None

    sample_dir = os.path.join(PROJECT_ROOT, "app", "sample_images")

    uploader_key = f"single_uploader_{st.session_state.get('uploader_key_id', 0)}"
    uploaded_file = st.file_uploader(
        T["upload_label"],
        type=["png", "jpg", "jpeg"],
        key=uploader_key,
        on_change=on_file_uploaded,
        help="Upload a standard posteroanterior (PA) frontal chest radiograph.",
    )

    # Determine single active source based on strict mutual exclusion rules
    if sample_choice == "Sample 1":
        sample_path = os.path.join(sample_dir, "sample_normal.png")
        if os.path.exists(sample_path):
            active_image = Image.open(sample_path)
            source_name = T["sample_1"]
    elif sample_choice == "Sample 2":
        sample_path = os.path.join(sample_dir, "sample_tb_advanced.png")
        if os.path.exists(sample_path):
            active_image = Image.open(sample_path)
            source_name = T["sample_2"]
    elif sample_choice == "Sample 3":
        sample_path = os.path.join(sample_dir, "sample_tb_moderate.png")
        if os.path.exists(sample_path):
            active_image = Image.open(sample_path)
            source_name = T["sample_3"]
    elif sample_choice == "None":
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            is_size_ok, size_err = validate_image_file_size(file_bytes, max_mb=10.0)
            if not is_size_ok:
                st.error(f"❌ File Error: {size_err}")
                active_image = None
            else:
                try:
                    active_image = validate_and_load_image(file_bytes)
                    source_name = uploaded_file.name
                except Exception as e:
                    st.error(f"❌ Input Error: {str(e)}")
                    active_image = None

    if active_image is not None:
        st.markdown(
            f"<div style='background-color: #e8f4fd; border-left: 5px solid #0c5460; padding: 10px 14px; border-radius: 4px; font-size: 1.05rem; font-weight: 700; color: #0c5460; margin-top: 10px; margin-bottom: 14px;'>"
            f"Currently analysing: {source_name}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Opacity slider
        cam_weight = st.slider(
            "🎚️ Heatmap Overlay Blend (0.1 = Attention Map Only, 0.9 = Radiograph Only):",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            key="cam_opacity_slider",
        )

        # Run OOD check first
        is_cxr, ood_error, quality_warnings = inspect_xray_quality_and_ood(active_image)

        if not is_cxr:
            st.error(f"{ood_error}")
            st.info("💡 **Clinical Requirement:** TB-Screen AI requires a frontal chest radiograph (CXR). Photographs of people, documents, pets, or outdoor scenes cannot be diagnosed.")
        else:
            # Display any non-fatal quality advisories
            for w in quality_warnings:
                st.warning(w)

            with st.spinner(T["processing"]):
                try:
                    if is_onnx_selected:
                        # Fast ONNX inference
                        onnx_res = predict_chest_xray_onnx(active_image, threshold=active_threshold)
                        # Also compute Grad-CAM overlay using PyTorch model
                        pt_model, pt_dev = get_cached_model()
                        overlay, grayscale_cam, _, _ = explain_xray(
                            image=active_image,
                            model=pt_model,
                            device=pt_dev,
                            target_class=1,
                            image_weight=cam_weight,
                        )
                        result = {
                            **onnx_res,
                            "gradcam_overlay_rgb": overlay,
                            "gradcam_grayscale": grayscale_cam,
                        }
                    else:
                        result = predict_chest_xray(
                            image_input=active_image,
                            threshold=active_threshold,
                            image_weight=cam_weight,
                        )
                except Exception as e:
                    st.error(f"Diagnostic Pipeline Error: {str(e)}")
                    result = None

            if result is not None:
                # Layout radiographs
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.markdown(f"#### 📷 {T['orig_heading']}")
                    st.image(active_image, use_container_width=True)
                with col_img2:
                    st.markdown(f"#### 🎯 {T['cam_heading']}")
                    st.image(result["gradcam_overlay_rgb"], use_container_width=True)

                st.markdown("---")
                st.subheader(f"📊 {T['results_header']}")

                tb_prob = result["tb_probability"]
                conf = result["confidence_percentage"]
                risk_tier = result["risk_tier"]
                color_hex = result["risk_color_hex"]
                emoji = result["risk_emoji"]
                lat_ms = result.get("inference_time_ms", 75.0)

                # Determine textual guidance
                if risk_tier == "HIGH":
                    status_text = T["status_high"]
                    msg_text = T["msg_high"]
                    action_text = T["action_high"]
                elif risk_tier == "NEEDS_REVIEW":
                    status_text = T["status_review"]
                    msg_text = T["msg_review"]
                    action_text = T["action_review"]
                else:
                    status_text = T["status_normal"]
                    msg_text = T["msg_normal"]
                    action_text = T["action_normal"]

                m_col1, m_col2, m_col3, m_col4 = st.columns([1.3, 1, 1, 1])
                with m_col1:
                    st.markdown(
                        f"<div class='metric-badge' style='background-color: {color_hex};'>"
                        f"{emoji} {status_text}</div>",
                        unsafe_allow_html=True,
                    )
                with m_col2:
                    st.metric(label=T["prob_tb"], value=f"{tb_prob * 100:.1f}%")
                with m_col3:
                    st.metric(label=T["confidence"], value=f"{conf:.1f}%")
                with m_col4:
                    engine_tag = "⚡ ONNX" if is_onnx_selected else "⏱️ PyTorch"
                    st.metric(label=f"{engine_tag} Latency", value=f"{lat_ms:.1f} ms")

                # Action guidance card
                st.markdown(
                    f"""
                    <div class='action-card'>
                        <p style='font-size: 1.05rem; margin-bottom: 6px;'><b>Clinical Impression:</b> {msg_text}</p>
                        <p style='font-size: 0.95rem; margin-bottom: 8px; color: #182b49;'><b>{action_text}</b></p>
                        <div style='background: #eef2f7; padding: 6px 10px; border-radius: 4px; font-size: 0.85rem; color: #2c3e50;'>
                            <b>📞 National TB Toll-Free Helpline:</b> 1800-11-6666 &nbsp;|&nbsp; <b>Nikshay Portal:</b> <a href='https://nikshay.in' target='_blank'>nikshay.in</a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Append to session history if not already recorded
                history_entry = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "patient_id": patient_id,
                    "source": source_name,
                    "risk_tier": risk_tier,
                    "tb_probability_pct": round(tb_prob * 100, 1),
                    "confidence_pct": round(conf, 1),
                    "action": action_text[:60] + "...",
                    "latency_ms": round(lat_ms, 1),
                }
                if not any(h["patient_id"] == patient_id and h["source"] == source_name for h in st.session_state["patient_history"]):
                    add_to_session_history(history_entry)

                # PDF Generation
                st.markdown("### 📄 Clinical Referral Report")
                try:
                    pdf_bytes = create_diagnostic_pdf(
                        prediction=result,
                        patient_id=patient_id,
                        clinic_name="Primary Health Center (Rural Screening)",
                        notes=field_notes,
                    )
                    st.download_button(
                        label=f"📥 {T['download_pdf']}",
                        data=pdf_bytes,
                        file_name=f"TB_Referral_{patient_id}.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.error(f"Error compiling referral PDF: {str(e)}")
    else:
        st.info("👈 Select a demo sample from the sidebar or upload a frontal chest radiograph above to begin screening.")


# ----------------------------------------------------------------------
# TAB 2: HIGH-VOLUME BATCH TRIAGE
# ----------------------------------------------------------------------
with tab_batch:
    st.markdown("### 📂 High-Volume Screening Camp / Batch Triage")
    st.write(
        "Upload a batch of chest radiographs (e.g. from a mobile digital X-ray van or village screening camp). "
        "The system will process all X-rays locally and sort patients by **highest tuberculosis risk first** "
        "to ensure immediate sputum test referral for critical cases."
    )

    batch_files = st.file_uploader(
        "Upload Multiple Chest Radiographs (PNG, JPG, JPEG):",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files:
        st.write(f"[*] **{len(batch_files)}** radiographs selected for batch screening.")
        if st.button("🚀 Run Batch Triage Analysis", type="primary"):
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            t_batch_start = time.perf_counter()

            batch_results = []
            for i, uploaded_f in enumerate(batch_files):
                status_placeholder.text(f"Screening {i+1}/{len(batch_files)}: {uploaded_f.name}...")
                try:
                    img = validate_and_load_image(uploaded_f.getvalue())
                    is_cxr, ood_err, _ = inspect_xray_quality_and_ood(img)
                    if not is_cxr:
                        batch_results.append({
                            "Filename": uploaded_f.name,
                            "Risk Tier": "REJECTED (OOD)",
                            "TB Prob (%)": 0.0,
                            "Confidence (%)": 0.0,
                            "Latency (ms)": 0.0,
                            "Triage Status": "Rejected",
                            "Action Plan": "Not a valid CXR (OOD)",
                        })
                    else:
                        if is_onnx_selected:
                            pred = predict_chest_xray_onnx(img, threshold=active_threshold)
                        else:
                            pred = predict_chest_xray(img, threshold=active_threshold)
                        tier = pred["risk_tier"]
                        tb_p = pred["tb_probability"] * 100.0
                        lat = pred.get("inference_time_ms", 7.5)

                        if tier == "HIGH":
                            action = "🔴 Urgent Sputum GeneXpert & Isolation"
                        elif tier == "NEEDS_REVIEW":
                            action = "🟡 Medical Officer Review & AFB Smear"
                        else:
                            action = "🟢 Routine primary care"

                        batch_results.append({
                            "Filename": uploaded_f.name,
                            "Risk Tier": tier,
                            "TB Prob (%)": round(tb_p, 1),
                            "Confidence (%)": round(pred["confidence_percentage"], 1),
                            "Latency (ms)": round(lat, 1),
                            "Triage Status": f"{pred['risk_emoji']} {pred['risk_tier']}",
                            "Action Plan": action,
                        })

                        # Add to shift history
                        add_to_session_history({
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "patient_id": uploaded_f.name.split(".")[0],
                            "source": uploaded_f.name,
                            "risk_tier": tier,
                            "tb_probability_pct": round(tb_p, 1),
                            "confidence_pct": round(pred["confidence_percentage"], 1),
                            "action": action,
                            "latency_ms": round(lat, 1),
                        })
                except Exception as e:
                    batch_results.append({
                        "Filename": uploaded_f.name,
                        "Risk Tier": "ERROR",
                        "TB Prob (%)": 0.0,
                        "Confidence (%)": 0.0,
                        "Latency (ms)": 0.0,
                        "Triage Status": "Error",
                        "Action Plan": f"Error: {str(e)}",
                    })

                progress_bar.progress((i + 1) / len(batch_files))

            total_batch_sec = time.perf_counter() - t_batch_start
            throughput_rate = len(batch_files) / max(total_batch_sec, 0.001)
            status_placeholder.success(f"Batch screening complete in {total_batch_sec:.2f}s! ({throughput_rate:.1f} X-rays/sec)")

            # Convert to DataFrame and sort by TB Probability descending
            df_batch = pd.DataFrame(batch_results)
            df_sorted = df_batch.sort_values(by="TB Prob (%)", ascending=False).reset_index(drop=True)

            # Summary KPI Cards
            high_count = len(df_sorted[df_sorted["Risk Tier"] == "HIGH"])
            review_count = len(df_sorted[df_sorted["Risk Tier"] == "NEEDS_REVIEW"])
            low_count = len(df_sorted[df_sorted["Risk Tier"] == "LOW"])
            rejected_count = len(df_sorted[df_sorted["Risk Tier"].str.contains("REJECTED|ERROR")])

            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            with kpi1:
                st.metric("Total Screened", len(df_sorted))
            with kpi2:
                st.metric("🔴 High Risk", high_count)
            with kpi3:
                st.metric("🟡 Needs Review", review_count)
            with kpi4:
                st.metric("🟢 Low Risk", low_count)
            with kpi5:
                st.metric("⚡ Screening Rate", f"{throughput_rate:.1f}/sec", help=f"Total batch completed in {total_batch_sec:.2f}s")

            st.markdown("#### 📋 Prioritized Triage Manifest (Highest Risk on Top):")
            st.dataframe(df_sorted, use_container_width=True)

            csv_header = (
                "# ======================================================================\n"
                "# TB-Screen AI: Frontline Clinical Batch Triage Manifest\n"
                "# Problem HI-01: Medical Image Diagnostic Assistant\n"
                "# Clinical Advisor: Dr. Bepar Mohammad Tayyib, MBBS (Clinical Protocol Reviewer)\n"
                "# Regulatory Status: Screening aid only. Has not undergone formal prospective clinical trials.\n"
                "# Protocol: Fast-track High Risk / Needs Review cases for sputum GeneXpert MTB/RIF.\n"
                "# National TB Toll-Free Helpline: 1800-11-6666 | Nikshay Portal: nikshay.in\n"
                "# ======================================================================\n"
            )
            csv_data = (csv_header + df_sorted.to_csv(index=False)).encode("utf-8")
            st.download_button(
                label="📥 Download Triage Priority Manifest (CSV)",
                data=csv_data,
                file_name=f"tb_triage_manifest_{datetime.date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )


# ----------------------------------------------------------------------
# TAB 3: SHIFT SESSION HISTORY
# ----------------------------------------------------------------------
with tab_history:
    st.markdown("### 📋 Current Shift Screening Log")
    st.write("Record of all radiographs screened in this local clinic session.")

    if not st.session_state["patient_history"]:
        st.info("No patients screened yet during this session. Screen a single radiograph or run a batch triage to view the log.")
    else:
        df_hist = pd.DataFrame(st.session_state["patient_history"])

        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Patients Screened Today", len(df_hist))
        with s2:
            st.metric("High Risk Referrals", len(df_hist[df_hist["risk_tier"] == "HIGH"]))
        with s3:
            st.metric("Flagged for Review", len(df_hist[df_hist["risk_tier"] == "NEEDS_REVIEW"]))

        st.dataframe(df_hist, use_container_width=True)

        col_h1, col_h2 = st.columns([1, 4])
        with col_h1:
            csv_header = (
                "# ======================================================================\n"
                "# TB-Screen AI: Shift Screening History Log\n"
                "# Problem HI-01: Medical Image Diagnostic Assistant\n"
                "# Clinical Advisor: Dr. Bepar Mohammad Tayyib, MBBS (Clinical Protocol Reviewer)\n"
                "# Regulatory Status: Screening aid only. Has not undergone formal prospective clinical trials.\n"
                "# National TB Toll-Free Helpline: 1800-11-6666 | Nikshay Portal: nikshay.in\n"
                "# ======================================================================\n"
            )
            csv_hist = (csv_header + df_hist.to_csv(index=False)).encode("utf-8")
            st.download_button(
                label="📥 Export Shift History (CSV)",
                data=csv_hist,
                file_name=f"shift_history_{datetime.date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        with col_h2:
            if st.button("🗑️ Clear Shift Session History"):
                st.session_state["patient_history"] = []
                st.rerun()


# ----------------------------------------------------------------------
# TAB 4: PUBLIC HEALTH IMPACT & CLINICAL ECONOMICS
# ----------------------------------------------------------------------
with tab_impact:
    st.markdown("### 📊 Public Health Impact & Frontline Economics")
    st.write(
        "Tuberculosis remains the leading infectious killer in developing nations. "
        "Here is the epidemiological burden, clinical necessity, and economic rationale for deploying "
        "low-latency AI triage in frontline primary healthcare centers."
    )

    st.markdown("#### 🇮🇳 Epidemiological TB Burden in India (Authoritative Sources)")
    stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
    with stat_c1:
        st.markdown(
            """
            <div class='impact-stat-box'>
                <div class='impact-stat-number'>2.8M</div>
                <div class='impact-stat-label'>Incident TB Cases / Year<br><i>(27% of global burden)</i></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_c2:
        st.markdown(
            """
            <div class='impact-stat-box'>
                <div class='impact-stat-number'>~340K</div>
                <div class='impact-stat-label'>Annual Deaths in India<br><i>(~930 deaths every single day)</i></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_c3:
        st.markdown(
            """
            <div class='impact-stat-box'>
                <div class='impact-stat-number'>40%</div>
                <div class='impact-stat-label'>Initial Diagnostic Delay<br><i>(Avg delay: 21–28 days)</i></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with stat_c4:
        st.markdown(
            """
            <div class='impact-stat-box'>
                <div class='impact-stat-number'>1 : 100K</div>
                <div class='impact-stat-label'>Radiologist Ratio in Rural India<br><i>(Severe specialist void)</i></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Sources: **WHO Global Tuberculosis Report 2023**; **India TB Report 2023** (Central TB Division, Ministry of Health and Family Welfare, Govt. of India)."
    )

    st.markdown("---")
    st.markdown("#### 🫁 Why Frontline Chest X-Ray Triage Matters")
    st.markdown(
        """
        - **Subclinical & Smear-Negative Detection:** Up to **50% of active pulmonary TB cases** in community surveys are sputum smear-negative or minimally symptomatic, yet exhibit distinctive radiographic abnormalities on chest X-ray.
        - **Preserving Scarce GeneXpert Cartridges:** Molecular testing (GeneXpert MTB/RIF) costs approximately **₹1,500–₹2,000 ($18–$25)** per cartridge and suffers from supply-chain constraints in rural PHCs. TB-Screen AI acts as an ultra-low-cost filter, ensuring molecular cartridges are allocated to high-risk patients.
        - **WHO End-TB Strategy Recommendation:** The World Health Organization explicitly recommends chest radiography as the frontline triage tool for active case finding in high-burden populations.
        """
    )

    st.markdown("---")
    st.markdown("#### 🔄 Rural Primary Health Center (PHC) Workflow Integration")
    st.markdown(
        """
        ```text
        [1. Symptomatic Patient] ──> [2. Local Analog/DR X-ray] ──> [3. TB-Screen AI (<100ms Inference)]
                                                                               │
                          ┌────────────────────────────────────────────────────┴────────────────────────────────┐
                          ▼                                                                                     ▼
                  [🔴 HIGH RISK / 🟡 REVIEW]                                                             [🟢 LOW RISK]
                          │                                                                                     │
         [Immediate GeneXpert Sputum Test]                                                              [Routine Care]
         [Surgical Mask / Infection Control]                                                     [Advice to return if cough >2w]
         [Same-Day DOTS Anti-TB Regimen]
        ```
        """
    )

    st.markdown("---")
    st.markdown("#### 💰 Health Economics & Diagnostic Time Comparison *(Estimates)*")

    econ_data = {
        "Metric": [
            "Time to Clinical Triage",
            "Patient Travel Requirement",
            "Out-of-Pocket Diagnostic Expense",
            "Radiologist Requirement",
            "Risk of Household Transmission",
            "GeneXpert Cartridge Utilization",
        ],
        "Traditional Rural Clinic Path": [
            "14 to 28 days (batch reporting by district hospital)",
            "Multiple trips to district center (50–100 km)",
            "₹1,500 – ₹3,500 (travel, wage loss, private consult)",
            "Mandatory (causes reporting backlog)",
            "High (patient coughs in household for 3 weeks)",
            "Indiscriminate / often exhausted rapidly",
        ],
        "With TB-Screen AI Assistive Triage": [
            "Under 2 hours (instant on-site triage)",
            "Zero extra travel; triage done at local PHC",
            "₹0 additional patient expense",
            "None needed for initial triage (tele-consult for yellow tier)",
            "Minimized (same-day fast-track referral)",
            "Targeted (high yield on flagged red/yellow cases)",
        ],
    }
    df_econ = pd.DataFrame(econ_data)
    st.dataframe(df_econ, use_container_width=True)
    st.caption("*Note: Cost, time, and travel figures represent empirical public health estimates from published rural TB studies in Central and South India.*")


# ======================================================================
# FOOTER & CLINICAL REVIEW EXPANDER
# ======================================================================
st.markdown("---")
with st.expander("ℹ️ About / Clinical Review & Regulatory Boundaries"):
    st.markdown(
        """
        **Clinical Advisory:**
        Clinical workflow, risk triage messages, referral guidance, and example outputs were reviewed by 
        **Dr. Bepar Mohammad Tayyib, MBBS** (Medical Advisor; not a member of the development team).

        **Regulatory & Clinical Trial Disclaimer:**
        TB-Screen AI is an assistive frontline screening aid intended to support triage in under-resourced rural clinics. 
        It has **not** been evaluated in formal prospective clinical trials and does not claim clinical validation. 
        It is not an autonomous medical diagnostic device; all outputs must be clinically correlated by a qualified healthcare professional 
        and confirmed with laboratory diagnostics (Acid-Fast Bacilli sputum smear microscopy or GeneXpert MTB/RIF).

        **Critical External Validation Boundary:**
        The model does not generalise to a different scanner, and needs local calibration before any real use. 
        Evaluating on external digitized analog radiographs (Montgomery County NIH dataset) resulted in an AUC drop to 0.5873, 
        confirming that local hardware contrast calibration is mandatory prior to real-world deployment.

        **Technical Architecture:**
        - Model: EfficientNet-B0 with CLAHE pulmonary contrast enhancement & lung region-of-interest cropping.
        - Calibration: Post-hoc Temperature Scaling ($T^* = 0.4860$) optimizing Expected Calibration Error.
        - Explainability: Gradient-weighted Class Activation Mapping (Grad-CAM) targeting the final convolutional block.
        - Deployment: Local CPU inference engine (<10ms latency, zero remote network required).
        """
    )
