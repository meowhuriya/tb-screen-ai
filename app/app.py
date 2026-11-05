"""TB-Screen AI: Medical Image Diagnostic Assistant (Hospital-Grade Clinical UI).

Track: HI-01 - Medical Image Diagnostic Assistant
Frontline clinical screening software for rural primary healthcare centers:
- Inter font, 8px grid, accessible WCAG AA contrast
- Clean hospital color system (Teal #0E6E8C, Slate #1B2733, Surface #FFFFFF)
- Dedicated risk colors used strictly for risk status
- Zero emojis; clean clinical badges and radiology viewer panels
- 100% offline edge execution (<10 ms CPU inference)
- Deterministic OOD safety filters & radiographic quality checks
- Trilingual support: English, Hindi, and Telugu
"""

import datetime
import io
import os
import sys
import time
from typing import Any, Dict, List

# Ensure project root and app directory are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_DIR = os.path.abspath(os.path.dirname(__file__))
for p in [PROJECT_ROOT, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from pdf_generator import create_diagnostic_pdf
except (ImportError, ModuleNotFoundError):
    from app.pdf_generator import create_diagnostic_pdf

try:
    from styles import inject_custom_css
except (ImportError, ModuleNotFoundError):
    from app.styles import inject_custom_css
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


_ = preload_inference_backends()

# Page configuration
st.set_page_config(
    page_title="TB-Screen AI | Frontline Diagnostic Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
)

# Inject hospital-grade design system
inject_custom_css()

# Multi-language dictionary
STRINGS: Dict[str, Dict[str, str]] = {
    "English": {
        "title": "TB-Screen AI",
        "subtitle": "Frontline pulmonary screening for rural primary care (HI-01)",
        "aid_badge": "Screening aid - not a diagnosis",
        "privacy_title": "Privacy and clinical notes",
        "privacy_text": (
            "Images are processed ephemerally in RAM and are not saved by the application. "
            "For full privacy, run locally without internet connection. "
            "The hosted web demonstration should not receive identifiable patient data. "
            "This software is an assistive screening aid; all findings must be confirmed with sputum smear microscopy or GeneXpert MTB/RIF."
        ),
        "select_lang": "Language / भाषा / భాష",
        "preset_title": "Demonstration Samples",
        "sample_1": "Normal Chest",
        "sample_2": "Advanced TB",
        "sample_3": "Moderate TB",
        "clear_selection": "Clear Selection",
        "upload_label": "Upload Frontal Chest Radiograph (PNG, JPG, JPEG):",
        "empty_state": "Choose a sample or upload a frontal chest X-ray to begin.",
        "analysing": "Analysing",
        "processing": "Analysing radiograph...",
        "orig_heading": "Original Chest Radiograph",
        "cam_heading": "Grad-CAM Pulmonary Saliency",
        "status_normal": "Low risk. No TB pattern detected by the model.",
        "msg_normal": "Low risk. No TB pattern detected by the model.",
        "action_normal": "Next step: Routine primary care. Advise patient to return if persistent cough (>2 weeks), fever, or night sweats develop.",
        "status_review": "Needs review. Findings are indeterminate; clinician review advised.",
        "msg_review": "Needs review. Findings are indeterminate; clinician review advised.",
        "action_review": "Next step: Review by Medical Officer. Collect sputum for acid-fast smear or GeneXpert MTB/RIF assay. National TB Helpline: 1800-11-6666.",
        "status_high": "High risk. TB suspected; confirm with a sputum test (GeneXpert or smear).",
        "msg_high": "High risk. TB suspected; confirm with a sputum test (GeneXpert or smear).",
        "action_high": "Next step: Fast-track sputum GeneXpert test and refer to DOTS clinic. Provide surgical mask. National TB Helpline: 1800-11-6666 | nikshay.in",
        "download_pdf": "Download Clinical Referral PDF",
        "patient_id_label": "Patient Identifier:",
        "notes_label": "Field observations / notes:",
        "tab_single": "Single Patient Screening",
        "tab_batch": "Batch Triage",
        "tab_history": "Session History",
        "tab_impact": "Impact & Economics",
    },
    "हिन्दी (Hindi)": {
        "title": "टीबी-स्क्रीन एआई",
        "subtitle": "ग्रामीण प्राथमिक स्वास्थ्य केंद्रों के लिए फुफ्फुसीय टीबी स्क्रीनिंग (HI-01)",
        "aid_badge": "स्क्रीनिंग सहायता - अंतिम निदान नहीं",
        "privacy_title": "गोपनीयता एवं चिकित्सीय दिशानिर्देश",
        "privacy_text": (
            "एक्स-रे चित्र केवल रैम में संसाधित किए जाते हैं और सर्वर पर सहेजे नहीं जाते। "
            "पूर्ण गोपनीयता के लिए इसे स्थानीय रूप से बिना इंटरनेट के चलाएं। "
            "होस्ट किए गए डेमो में मरीज का व्यक्तिगत डेटा अपलोड न करें। "
            "यह केवल स्क्रीनिंग सहायता है; सभी निष्कर्षों की पुष्टि थूक जांच या GeneXpert परीक्षण द्वारा अनिवार्य है।"
        ),
        "select_lang": "Language / भाषा / భాష",
        "preset_title": "डेमो नमूने",
        "sample_1": "सामान्य चेस्ट",
        "sample_2": "गंभीर टीबी",
        "sample_3": "मध्यम टीबी",
        "clear_selection": "चयन हटाएं",
        "upload_label": "चेस्ट एक्स-रे अपलोड करें (PNG, JPG, JPEG):",
        "empty_state": "स्क्रीनिंग शुरू करने के लिए कोई नमूना चुनें या एक्स-रे अपलोड करें।",
        "analysing": "विश्लेषण जारी",
        "processing": "एक्स-रे का विश्लेषण किया जा रहा है...",
        "orig_heading": "मूल चेस्ट एक्स-रे",
        "cam_heading": "ग्रैड-सीएएम हीटमैप",
        "status_normal": "कम जोखिम। मॉडल द्वारा टीबी का कोई पैटर्न नहीं पाया गया।",
        "msg_normal": "कम जोखिम। मॉडल द्वारा टीबी का कोई पैटर्न नहीं पाया गया।",
        "action_normal": "अगला कदम: सामान्य प्राथमिक उपचार। यदि 2 सप्ताह से अधिक खांसी, बुखार या पसीना आए तो दोबारा दिखाएं।",
        "status_review": "समीक्षा आवश्यक। निष्कर्ष अनिर्णीत हैं; चिकित्सक द्वारा समीक्षा की सलाह दी जाती है।",
        "msg_review": "समीक्षा आवश्यक। निष्कर्ष अनिर्णीत हैं; चिकित्सक द्वारा समीक्षा की सलाह दी जाती है।",
        "action_review": "अगला कदम: चिकित्सा अधिकारी से समीक्षा कराएं। थूक की जांच (AFB स्मीयर) या GeneXpert परीक्षण कराएं। राष्ट्रीय टीबी हेल्पलाइन: 1800-11-6666।",
        "status_high": "उच्च जोखिम। टीबी की आशंका; थूक परीक्षण (GeneXpert या स्मीयर) से पुष्टि करें।",
        "msg_high": "उच्च जोखिम। टीबी की आशंका; थूक परीक्षण (GeneXpert या स्मीयर) से पुष्टि करें।",
        "action_high": "अगला कदम: मरीज को मास्क पहनाएं। तत्काल GeneXpert जांच करवाएं और डॉट्स (DOTS) केंद्र भेजें। राष्ट्रीय टीबी हेल्पलाइन: 1800-11-6666 | nikshay.in",
        "download_pdf": "क्लिनिकल रेफरल पीडीएफ डाउनलोड करें",
        "patient_id_label": "मरीज पहचान संख्या:",
        "notes_label": "स्वास्थ्य कार्यकर्ता की टिप्पणी:",
        "tab_single": "एकल स्क्रीनिंग",
        "tab_batch": "बैच ट्राइएज",
        "tab_history": "सत्र इतिहास",
        "tab_impact": "प्रभाव व बचत",
    },
    "తెలుగు (Telugu)": {
        "title": "టిబి-స్క్రీన్ AI",
        "subtitle": "గ్రామీణ ప్రాథమిక ఆరోగ్య కేంద్రాల కోసం క్షయవ్యాధి స్క్రీనింగ్ (HI-01)",
        "aid_badge": "స్క్రీనింగ్ సహాయం - తుది నిర్ధారణ కాదు",
        "privacy_title": "గోప్యత మరియు క్లినికల్ మార్గదర్శకాలు",
        "privacy_text": (
            "చిత్రాలు మెమరీలో మాత్రమే విశ్లేషించబడతాయి మరియు భద్రపరచబడవు. "
            "పూర్తి గోప్యత కోసం ఇంటర్నెట్ లేకుండా స్థానికంగా ఉపయోగించండి. "
            "లైవ్ డెమోలో రోగుల వ్యక్తిగత వివరాలను నమోదు చేయవద్దు. "
            "ఇది కేవలం ప్రాథమిక స్క్రీనింగ్ సహాయకం మాత్రమే; నిర్ధారణ కోసం కఫం లేదా GeneXpert పరీక్ష తప్పనిసరి."
        ),
        "select_lang": "Language / भाषा / భాష",
        "preset_title": "నమూనా ఎక్స్-రేలు",
        "sample_1": "సాధారణ ఛాతీ",
        "sample_2": "తీవ్రమైన క్షయ",
        "sample_3": "మధ్యస్థాయి క్షయ",
        "clear_selection": "ఎంపికను తొలగించు",
        "upload_label": "ఛాతీ ఎక్స్-రే అప్‌లోడ్ చేయండి (PNG, JPG, JPEG):",
        "empty_state": "ప్రారంభించడానికి నమూనాను ఎంచుకోండి లేదా ఎక్స్-రే అప్‌లోడ్ చేయండి.",
        "analysing": "విశ్లేషిస్తోంది",
        "processing": "ఎక్స్-రే విశ్లేషించబడుతోంది...",
        "orig_heading": "అసలు ఛాతీ ఎక్స్-రే",
        "cam_heading": "గ్రాడ్-కామ్ హీట్‌మ్యాప్",
        "status_normal": "తక్కువ ప్రమాదం. మోడల్ ద్వారా ఎటువంటి క్షయ (TB) నమూనా గుర్తించబడలేదు.",
        "msg_normal": "తక్కువ ప్రమాదం. మోడల్ ద్వారా ఎటువంటి క్షయ (TB) నమూనా గుర్తించబడలేదు.",
        "action_normal": "తదుపరి చర్య: సాధారణ ఆరోగ్య సంరక్షణ. 2 వారాల కంటే ఎక్కువ దగ్గు లేదా జ్వరం ఉంటే సంప్రదించండి.",
        "status_review": "సమీక్ష అవసరం. ఫలితాలు అనిశ్చితంగా ఉన్నాయి; వైద్యుని సమీక్ష సూచించబడింది.",
        "msg_review": "సమీక్ష అవసరం. ఫలితాలు అనిశ్చితంగా ఉన్నాయి; వైద్యుని సమీక్ష సూచించబడింది.",
        "action_review": "తదుపరి చర్య: వైద్యాధికారిచే పరీక్షించండి. కఫం పరీక్ష లేదా GeneXpert చేయించండి. జాతీయ టిబి హెల్ప్‌లైన్: 1800-11-6666.",
        "status_high": "అధిక ప్రమాదం. క్షయ (TB) అనుమానించబడింది; కఫం పరీక్ష (GeneXpert లేదా స్మియర్) ద్వారా నిర్ధారించండి.",
        "msg_high": "అధిక ప్రమాదం. క్షయ (TB) అనుమానించబడింది; కఫం పరీక్ష (GeneXpert లేదా స్మియర్) ద్వారా నిర్ధారించండి.",
        "action_high": "తదుపరి చర్య: రోగికి మాస్క్ ధరింపజేయండి. వెంటనే GeneXpert పరీక్ష మరియు డాట్స్ (DOTS) కేంద్రానికి పంపండి. జాతీయ టిబి హెల్ప్‌లైన్: 1800-11-6666 | nikshay.in",
        "download_pdf": "క్లినికల్ రిఫెరల్ PDF డౌన్‌లోడ్ చేయండి",
        "patient_id_label": "రోగి గుర్తింపు సంఖ్య:",
        "notes_label": "ఫీల్డ్ వర్కర్ గమనికలు:",
        "tab_single": "ఒంటరి స్క్రీనింగ్",
        "tab_batch": "బ్యాచ్ ట్రయేజ్",
        "tab_history": "సెషన్ చరిత్ర",
        "tab_impact": "ఆర్థిక ప్రభావం",
    },
}

# Session state initialization
if "patient_history" not in st.session_state:
    st.session_state["patient_history"] = []

if "sample_selection" not in st.session_state:
    st.session_state["sample_selection"] = "None"

if "uploader_key_id" not in st.session_state:
    st.session_state["uploader_key_id"] = 0

if "patient_id" not in st.session_state:
    st.session_state["patient_id"] = f"PAT-{datetime.date.today().strftime('%Y%m%d')}-01"

if "patient_notes" not in st.session_state:
    st.session_state["patient_notes"] = ""


def select_sample(sample_val: str) -> None:
    """Callback to switch sample selection before widgets instantiate."""
    st.session_state["sample_selection"] = sample_val
    if sample_val != "None":
        st.session_state["uploader_key_id"] = st.session_state.get("uploader_key_id", 0) + 1


def add_to_session_history(record: Dict[str, Any]) -> None:
    """Store patient screening record in browser session."""
    st.session_state["patient_history"].append(record)


def on_sample_radio_changed() -> None:
    """Callback when user selects a demo sample."""
    choice = st.session_state.get("sample_selection", "None")
    if choice != "None":
        st.session_state["uploader_key_id"] = st.session_state.get("uploader_key_id", 0) + 1


def on_file_uploaded() -> None:
    """Callback when a file is uploaded."""
    current_key = f"single_uploader_{st.session_state.get('uploader_key_id', 0)}"
    uploaded = st.session_state.get(current_key)
    if uploaded is not None:
        st.session_state["sample_selection"] = "None"


# ======================================================================
# SIDEBAR (Minimal & Clean: Language & Preset Radio for Test Harness)
# ======================================================================
with st.sidebar:
    st.markdown("<div class='section-title'>Settings</div>", unsafe_allow_html=True)

    selected_lang_key = st.selectbox(
        STRINGS["English"]["select_lang"],
        options=["English", "हिन्दी (Hindi)", "తెలుగు (Telugu)"],
        index=0,
        key="app_language_selector",
    )
    T = STRINGS[selected_lang_key]

    st.markdown("---")
    st.markdown("<div class='caption-muted'><b>Operating Threshold</b></div>", unsafe_allow_html=True)
    threshold_mode = st.radio(
        "Operating Threshold:",
        options=[
            "WHO Benchmark (τ = 0.05) [Default]",
            "Calibrated Cutoff (τ = 0.27)",
        ],
        index=0,
        label_visibility="collapsed",
    )
    if "0.05" in threshold_mode:
        active_threshold = WHO_BENCHMARK_THRESHOLD
        threshold_label = "WHO Benchmark (τ = 0.05)"
    else:
        active_threshold = CALIBRATED_THRESHOLD
        threshold_label = "Calibrated Cutoff (τ = 0.27)"

    st.markdown("---")
    st.markdown("<div class='caption-muted'><b>Inference Engine</b></div>", unsafe_allow_html=True)
    engine_choice = st.radio(
        "Inference Engine:",
        options=[
            "PyTorch with Grad-CAM, batch",
            "ONNX Runtime, single image, local CPU (7.19 ms)",
        ],
        index=0,
        label_visibility="collapsed",
    )
    is_onnx_selected = "ONNX" in engine_choice


# ======================================================================
# HELPER FUNCTIONS (Layout and Components)
# ======================================================================
def render_patient_strip(patient_id_val: str, timestamp_str: str, img_name: str, thresh_str: str) -> None:
    """Render slim report-header patient metadata strip."""
    st.markdown(
        f"""
        <div class="patient-strip">
            <div class="patient-strip-item">
                <span class="patient-strip-label">Patient ID</span>
                <span class="patient-strip-val">{patient_id_val}</span>
            </div>
            <div class="patient-strip-item">
                <span class="patient-strip-label">Screening Timestamp</span>
                <span class="patient-strip-val">{timestamp_str}</span>
            </div>
            <div class="patient-strip-item">
                <span class="patient-strip-label">Radiograph Image</span>
                <span class="patient-strip-val">{img_name}</span>
            </div>
            <div class="patient-strip-item">
                <span class="patient-strip-label">Active Threshold</span>
                <span class="patient-strip-val">{thresh_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(pred: Dict[str, Any], active_thresh: float, lang_dict: Dict[str, str]) -> None:
    """Render primary clinical result card with risk badge, probability bar, and action plan."""
    risk_tier = pred["risk_tier"]
    tb_prob = pred["tb_probability"]
    prob_pct = tb_prob * 100.0
    thresh_pct = active_thresh * 100.0

    if risk_tier == "HIGH":
        badge_cls = "badge-high-risk"
        status_label = lang_dict["status_high"]
        next_step_text = lang_dict["action_high"]
        fill_cls = "prob-fill-high"
        icon_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#B42318" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'
    elif risk_tier == "NEEDS_REVIEW":
        badge_cls = "badge-review"
        status_label = lang_dict["status_review"]
        next_step_text = lang_dict["action_review"]
        fill_cls = "prob-fill-review"
        icon_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#9A6700" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
    else:
        badge_cls = "badge-low-risk"
        status_label = lang_dict["status_normal"]
        next_step_text = lang_dict["action_normal"]
        fill_cls = "prob-fill-low"
        icon_svg = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#1E7B4F" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>'

    # Scanner shift detection tag
    shift_res = pred.get("shift_detection")
    shift_tag_html = ""
    if shift_res is not None:
        if shift_res.get("is_unfamiliar"):
            shift_tag_html = '<div class="scanner-tag-unfamiliar">Unfamiliar scanner profile - local calibration recommended</div>'
        else:
            shift_tag_html = '<div class="scanner-tag-familiar">Familiar scanner profile</div>'

    fill_width = min(max(prob_pct, 1.0), 100.0)
    tick_pos = min(max(thresh_pct, 0.0), 100.0)

    st.markdown(
        f"""
        <div class="clinical-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
                <div class="result-status-badge {badge_cls}">
                    {icon_svg}
                    <span>{status_label}</span>
                </div>
                {shift_tag_html}
            </div>
            <div class="probability-label">Estimated Tuberculosis Probability</div>
            <div class="probability-display">{prob_pct:.1f}%</div>
            <div class="prob-track">
                <div class="{fill_cls}" style="width: {fill_width:.1f}%;"></div>
                <div class="threshold-tick" style="left: {tick_pos:.1f}%;" title="Decision Threshold: {thresh_pct:.1f}%"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 11.5px; color: #5B6B7B; margin-top: -10px; margin-bottom: 14px;">
                <span>0%</span>
                <span>Cutoff: {thresh_pct:.1f}%</span>
                <span>100%</span>
            </div>
            <div class="next-step-banner">
                <b>Action Plan:</b> {next_step_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_viewer_panel(orig_image: Image.Image, cam_image: np.ndarray, lang_dict: Dict[str, str]) -> float:
    """Render radiology viewer dark panels and opacity blend slider."""
    st.markdown("<div class='section-title'>Radiographic & Attention Viewer</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="radiology-box">
                <div class="radiology-caption">{lang_dict['orig_heading']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(orig_image, use_container_width=True)
    with c2:
        st.markdown(
            f"""
            <div class="radiology-box">
                <div class="radiology-caption">{lang_dict['cam_heading']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(cam_image, use_container_width=True)
        st.markdown(
            """
            <div class="legend-bar">
                <div class="legend-gradient"></div>
                <div class="legend-labels">
                    <span>Lower attention</span>
                    <span>Higher attention</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    opacity = st.slider(
        "Heatmap overlay opacity",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        key="cam_opacity_slider",
        help="Adjust balance between visual attention overlay and underlying anatomical parenchyma.",
    )
    return opacity


# ======================================================================
# MAIN CONTENT LAYOUT
# ======================================================================

# 1. Header Row
col_h_left, col_h_right = st.columns([3, 1])
with col_h_left:
    st.markdown(f"<div class='app-title'>{T['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='app-subtitle'>{T['subtitle']}</div>", unsafe_allow_html=True)
with col_h_right:
    st.markdown(
        f"<div style='text-align: right; padding-top: 6px;'><span class='header-aid-badge'>{T['aid_badge']}</span></div>",
        unsafe_allow_html=True,
    )

# Header Privacy Expander (closed by default)
with st.expander(T["privacy_title"], expanded=False):
    st.markdown(f"<div class='caption-muted'>{T['privacy_text']}</div>", unsafe_allow_html=True)

# Main Navigation Tabs (Plain text, no emojis)
tab_single, tab_batch, tab_history, tab_impact = st.tabs([
    T["tab_single"],
    T["tab_batch"],
    T["tab_history"],
    T["tab_impact"],
])


# ----------------------------------------------------------------------
# TAB 1: SINGLE PATIENT SCREENING
# ----------------------------------------------------------------------
with tab_single:
    sample_dir = os.path.join(PROJECT_ROOT, "app", "sample_images")

    # 2. Input Card: Sample buttons with thumbnails + Upload box
    st.markdown("<div class='section-title'>Select Radiograph Source</div>", unsafe_allow_html=True)
    input_c1, input_c2 = st.columns([1.1, 1.0])

    with input_c1:
        st.markdown("<div class='caption-muted' style='margin-bottom: 8px;'><b>Demonstration Benchmarks:</b></div>", unsafe_allow_html=True)
        col_s1, col_s2, col_s3 = st.columns(3)

        # Thumbnails
        s1_path = os.path.join(sample_dir, "sample_normal.png")
        s2_path = os.path.join(sample_dir, "sample_tb_advanced.png")
        s3_path = os.path.join(sample_dir, "sample_tb_moderate.png")

        with col_s1:
            if os.path.exists(s1_path):
                thumb_1 = Image.open(s1_path).resize((70, 70))
                st.image(thumb_1, width=70)
            st.button(T["sample_1"], key="btn_s1", on_click=select_sample, args=("Sample 1",), use_container_width=True)

        with col_s2:
            if os.path.exists(s2_path):
                thumb_2 = Image.open(s2_path).resize((70, 70))
                st.image(thumb_2, width=70)
            st.button(T["sample_2"], key="btn_s2", on_click=select_sample, args=("Sample 2",), use_container_width=True)

        with col_s3:
            if os.path.exists(s3_path):
                thumb_3 = Image.open(s3_path).resize((70, 70))
                st.image(thumb_3, width=70)
            st.button(T["sample_3"], key="btn_s3", on_click=select_sample, args=("Sample 3",), use_container_width=True)

        if st.session_state.get("sample_selection") != "None":
            st.button(T["clear_selection"], key="btn_clear", on_click=select_sample, args=("None",), use_container_width=True)

    with input_c2:
        st.markdown("<div class='caption-muted' style='margin-bottom: 8px;'><b>Custom Patient Upload:</b></div>", unsafe_allow_html=True)
        uploader_key = f"single_uploader_{st.session_state.get('uploader_key_id', 0)}"
        uploaded_file = st.file_uploader(
            T["upload_label"],
            type=["png", "jpg", "jpeg"],
            key=uploader_key,
            on_change=on_file_uploaded,
            label_visibility="collapsed",
        )

    # Determine single active source
    sample_choice = st.session_state.get("sample_selection", "None")
    active_image = None
    source_name = None

    if sample_choice == "Sample 1":
        if os.path.exists(s1_path):
            active_image = Image.open(s1_path)
            source_name = f"sample_normal.png ({T['sample_1']})"
    elif sample_choice == "Sample 2":
        if os.path.exists(s2_path):
            active_image = Image.open(s2_path)
            source_name = f"sample_tb_advanced.png ({T['sample_2']})"
    elif sample_choice == "Sample 3":
        if os.path.exists(s3_path):
            active_image = Image.open(s3_path)
            source_name = f"sample_tb_moderate.png ({T['sample_3']})"
    elif sample_choice == "None":
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            is_size_ok, size_err = validate_image_file_size(file_bytes, max_mb=10.0)
            if not is_size_ok:
                st.error(f"File invalid: {size_err}")
                active_image = None
            else:
                try:
                    active_image = validate_and_load_image(file_bytes)
                    source_name = uploaded_file.name
                except Exception as e:
                    st.error(f"Input error: {str(e)}")
                    active_image = None

    # Status / Empty State
    if active_image is None:
        st.info(T["empty_state"])
    else:
        st.markdown(
            f"<div class='caption-muted' style='margin: 8px 0 16px 0;'>"
            f"Currently analysing: <b>{source_name}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # 3. Patient Strip (slim report-header style)
        now_str = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
        render_patient_strip(
            patient_id_val=st.session_state["patient_id"],
            timestamp_str=now_str,
            img_name=source_name,
            thresh_str=threshold_label,
        )

        with st.expander("Patient observations and field notes", expanded=False):
            col_id, col_note = st.columns([1, 2])
            with col_id:
                st.session_state["patient_id"] = st.text_input(
                    T["patient_id_label"],
                    value=st.session_state["patient_id"],
                )
            with col_note:
                st.session_state["patient_notes"] = st.text_input(
                    T["notes_label"],
                    value=st.session_state["patient_notes"],
                    placeholder="e.g. Cough > 2 weeks, evening fever, weight loss",
                )

        # Check OOD safety
        is_cxr, ood_error, quality_warnings = inspect_xray_quality_and_ood(active_image)

        if not is_cxr:
            st.error(
                "Invalid radiograph: The image does not meet standard grayscale chest X-ray criteria. "
                "Please upload a standard frontal chest radiograph (CXR)."
            )
        else:
            for w in quality_warnings:
                st.warning(w)

            # Inference with spinner
            with st.spinner(T["processing"]):
                try:
                    cam_weight_val = st.session_state.get("cam_opacity_slider", 0.5)
                    if is_onnx_selected:
                        onnx_res = predict_chest_xray_onnx(active_image, threshold=active_threshold)
                        pt_model, pt_dev = get_cached_model()
                        overlay, grayscale_cam, _, _ = explain_xray(
                            image=active_image,
                            model=pt_model,
                            device=pt_dev,
                            target_class=1,
                            image_weight=cam_weight_val,
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
                            image_weight=cam_weight_val,
                        )
                except Exception as e:
                    st.error(f"Inference error: {str(e)}")
                    result = None

            if result is not None:
                # 4. RESULT CARD (Main Element)
                render_result_card(result, active_threshold, T)

                # 5. Image Viewer Card
                _ = render_viewer_panel(active_image, result["gradcam_overlay_rgb"], T)

                # Append to session history if not already recorded
                history_entry = {
                    "timestamp": now_str,
                    "patient_id": st.session_state["patient_id"],
                    "source": source_name,
                    "risk_tier": result["risk_tier"],
                    "tb_probability_pct": round(result["tb_probability"] * 100, 1),
                    "confidence_pct": round(result["confidence_percentage"], 1),
                    "action": result["predicted_label"],
                    "latency_ms": round(result.get("inference_time_ms", 7.5), 1),
                }
                if not any(h["patient_id"] == st.session_state["patient_id"] and h["source"] == source_name for h in st.session_state["patient_history"]):
                    add_to_session_history(history_entry)

                st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

                # 6. Collapsed Expanders under viewer
                with st.expander("Decision threshold configuration", expanded=False):
                    st.markdown(
                        "<div class='caption-muted'>"
                        "<b>WHO Benchmark (τ = 0.05):</b> Recommended for frontline community screening camps. "
                        "Prioritizes high sensitivity (95.10% sensitivity; 97/102 active cases detected) to minimize false negatives, "
                        "with 86.29% specificity.<br>"
                        "<b>Calibrated Cutoff (τ = 0.27):</b> Balanced clinical operating point minimizing false alarms "
                        "(94.67% specificity) with 88.24% sensitivity."
                        "</div>",
                        unsafe_allow_html=True,
                    )

                with st.expander("Technical latency & execution details", expanded=False):
                    lat_ms = result.get("inference_time_ms", 7.19)
                    measured_label = "ONNX Runtime, single image, local CPU" if is_onnx_selected else "PyTorch with Grad-CAM, batch"
                    st.markdown(
                        f"<div class='caption-muted'>"
                        f"• Measured Mode: <b>{measured_label}</b><br>"
                        f"• Single-Image Latency: <b>{lat_ms:.2f} ms</b> (Source: results/onnx_benchmark.json)<br>"
                        f"• Peak Memory RSS: <b>532.9 MB</b> (Source: results/benchmark_results.json)<br>"
                        f"• Execution Mode: 100% Offline-capable, in-memory RAM processing"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Download PDF button
                try:
                    pdf_bytes = create_diagnostic_pdf(
                        prediction=result,
                        patient_id=st.session_state["patient_id"],
                        clinic_name="Primary Health Center (Rural Screening)",
                        notes=st.session_state["patient_notes"],
                    )
                    st.download_button(
                        label=T["download_pdf"],
                        data=pdf_bytes,
                        file_name=f"TB_Referral_{st.session_state['patient_id']}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Error compiling referral PDF: {str(e)}")


# ----------------------------------------------------------------------
# TAB 2: BATCH TRIAGE
# ----------------------------------------------------------------------
with tab_batch:
    st.markdown("<div class='section-title'>High-Volume Screening Camp / Batch Triage</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='caption-muted' style='margin-bottom: 16px;'>"
        "Upload a batch of chest radiographs from a mobile screening camp or clinic folder. "
        "The system processes all images locally and orders patients by highest risk first."
        "</div>",
        unsafe_allow_html=True,
    )

    batch_files = st.file_uploader(
        "Upload Multiple Radiographs:",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if batch_files:
        st.markdown(f"<div class='caption-muted'><b>{len(batch_files)}</b> radiographs selected.</div>", unsafe_allow_html=True)
        if st.button("Run Batch Triage Analysis", type="primary"):
            progress_bar = st.progress(0)
            status_box = st.empty()
            t_batch_start = time.perf_counter()

            batch_results = []
            for i, uploaded_f in enumerate(batch_files):
                status_box.text(f"Screening {i+1}/{len(batch_files)}: {uploaded_f.name}...")
                try:
                    img = validate_and_load_image(uploaded_f.getvalue())
                    is_cxr, ood_err, _ = inspect_xray_quality_and_ood(img)
                    if not is_cxr:
                        batch_results.append({
                            "Filename": uploaded_f.name,
                            "Risk Tier": "REJECTED",
                            "TB Prob (%)": 0.0,
                            "Confidence (%)": 0.0,
                            "Latency (ms)": 0.0,
                            "Triage Status": "Rejected (OOD)",
                            "Action Plan": "Not a valid CXR",
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
                            action = "Urgent Sputum GeneXpert & Isolation"
                        elif tier == "NEEDS_REVIEW":
                            action = "Medical Officer Review & AFB Smear"
                        else:
                            action = "Routine primary care"

                        batch_results.append({
                            "Filename": uploaded_f.name,
                            "Risk Tier": tier,
                            "TB Prob (%)": round(tb_p, 1),
                            "Confidence (%)": round(pred["confidence_percentage"], 1),
                            "Latency (ms)": round(lat, 1),
                            "Triage Status": tier,
                            "Action Plan": action,
                        })

                        add_to_session_history({
                            "timestamp": datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
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

            total_sec = time.perf_counter() - t_batch_start
            rate = len(batch_files) / max(total_sec, 0.001)
            status_box.markdown(
                f"<div class='caption-muted' style='margin-bottom: 12px; color: #1E7B4F; font-weight: 500;'>"
                f"Batch screening completed in {total_sec:.2f}s ({rate:.1f} X-rays/second)."
                f"</div>",
                unsafe_allow_html=True,
            )

            df_batch = pd.DataFrame(batch_results)
            df_sorted = df_batch.sort_values(by="TB Prob (%)", ascending=False).reset_index(drop=True)

            # Summary KPI row
            high_count = len(df_sorted[df_sorted["Risk Tier"] == "HIGH"])
            rev_count = len(df_sorted[df_sorted["Risk Tier"] == "NEEDS_REVIEW"])
            low_count = len(df_sorted[df_sorted["Risk Tier"] == "LOW"])

            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Total Screened", len(df_sorted))
            with k2:
                st.metric("High Risk", high_count)
            with k3:
                st.metric("Needs Review", rev_count)
            with k4:
                st.metric("Low Risk", low_count)

            st.dataframe(df_sorted, use_container_width=True)

            csv_header = (
                "# ======================================================================\n"
                "# TB-Screen AI: Frontline Clinical Batch Triage Manifest\n"
                "# Problem HI-01: Medical Image Diagnostic Assistant\n"
                "# Advisor: Dr. Bepar Mohammad Tayyib, MBBS (Clinical Protocol Reviewer)\n"
                "# National TB Helpline: 1800-11-6666 | nikshay.in\n"
                "# ======================================================================\n"
            )
            csv_bytes = (csv_header + df_sorted.to_csv(index=False)).encode("utf-8")
            st.download_button(
                label="Download Triage Manifest (CSV)",
                data=csv_bytes,
                file_name=f"tb_triage_manifest_{datetime.date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )


# ----------------------------------------------------------------------
# TAB 3: SHIFT HISTORY
# ----------------------------------------------------------------------
with tab_history:
    st.markdown("<div class='section-title'>Session Screening History</div>", unsafe_allow_html=True)
    if not st.session_state["patient_history"]:
        st.info("No radiographs screened during this session yet.")
    else:
        df_hist = pd.DataFrame(st.session_state["patient_history"])

        h1, h2, h3 = st.columns(3)
        with h1:
            st.metric("Screened This Session", len(df_hist))
        with h2:
            st.metric("High Risk Flagged", len(df_hist[df_hist["risk_tier"] == "HIGH"]))
        with h3:
            st.metric("Review Cases", len(df_hist[df_hist["risk_tier"] == "NEEDS_REVIEW"]))

        st.dataframe(df_hist, use_container_width=True)

        col_hist_btn1, col_hist_btn2 = st.columns([1, 4])
        with col_hist_btn1:
            csv_header = (
                "# ======================================================================\n"
                "# TB-Screen AI: Shift Screening History Log\n"
                "# Advisor: Dr. Bepar Mohammad Tayyib, MBBS\n"
                "# ======================================================================\n"
            )
            csv_hist = (csv_header + df_hist.to_csv(index=False)).encode("utf-8")
            st.download_button(
                label="Export Session Log (CSV)",
                data=csv_hist,
                file_name=f"tb_session_history_{datetime.date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        with col_hist_btn2:
            if st.button("Clear Session History"):
                st.session_state["patient_history"] = []
                st.rerun()


# ----------------------------------------------------------------------
# TAB 4: IMPACT & ECONOMICS
# ----------------------------------------------------------------------
with tab_impact:
    st.markdown("<div class='section-title'>Public Health Burden & Clinical Economics</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='caption-muted' style='margin-bottom: 20px;'>"
        "Epidemiological data and resource-efficiency rationale for deploying edge AI triage in rural primary healthcare centers."
        "</div>",
        unsafe_allow_html=True,
    )

    c_stat1, c_stat2, c_stat3, c_stat4 = st.columns(4)
    with c_stat1:
        st.markdown(
            """
            <div class="clinical-card" style="text-align: center; padding: 18px 12px;">
                <div style="font-size: 28px; font-weight: 700; color: #0E6E8C;">2.8M</div>
                <div class="caption-muted">Incident TB Cases / Year<br>(27% of global burden)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_stat2:
        st.markdown(
            """
            <div class="clinical-card" style="text-align: center; padding: 18px 12px;">
                <div style="font-size: 28px; font-weight: 700; color: #B42318;">~340K</div>
                <div class="caption-muted">Annual Deaths in India<br>(~930 deaths per day)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_stat3:
        st.markdown(
            """
            <div class="clinical-card" style="text-align: center; padding: 18px 12px;">
                <div style="font-size: 28px; font-weight: 700; color: #9A6700;">1 : 100K</div>
                <div class="caption-muted">Rural Radiologist Ratio<br>(Specialist shortage)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_stat4:
        st.markdown(
            """
            <div class="clinical-card" style="text-align: center; padding: 18px 12px;">
                <div style="font-size: 28px; font-weight: 700; color: #1E7B4F;">21–28d</div>
                <div class="caption-muted">Standard Reporting Delay<br>(Reduced to 7.19 ms on local CPU)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='caption-muted' style='margin: 8px 0 24px 0; font-size: 12px;'>"
        "Sources: WHO Global Tuberculosis Report 2023; Central TB Division, Ministry of Health & Family Welfare, Govt. of India."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Diagnostic Workflow & Cartridge Preservation</div>", unsafe_allow_html=True)
    econ_data = {
        "Metric": [
            "Time to Clinical Triage",
            "Patient Travel Requirement",
            "Out-of-Pocket Diagnostic Expense",
            "Specialist Bottleneck",
            "Household Transmission Window",
            "Molecular Cartridge Utilization",
        ],
        "Traditional Rural Pathway": [
            "14 to 28 days (remote batch radiologist reading)",
            "Multiple trips to district hospital (50–100 km)",
            "₹1,500 – ₹3,500 (travel, wage loss, private consultation)",
            "Severe shortage delays interpretation",
            "High (patient coughs in household during delay)",
            "Indiscriminate usage leads to stockouts",
        ],
        "With TB-Screen AI Edge Triage": [
            "Under 2 minutes (immediate on-site classification)",
            "Zero additional travel; screening completed at local PHC",
            "₹0 additional patient expense",
            "Assistive triage frees specialists for complex reviews",
            "Minimized (same-day sputum referral and isolation)",
            "Preserves scarce GeneXpert cartridges for flagged cases",
        ],
    }
    df_econ = pd.DataFrame(econ_data)
    st.dataframe(df_econ, use_container_width=True)


# ======================================================================
# 7. FOOTER
# ======================================================================
st.markdown(
    """
    <div class="clinic-footer">
        Clinical protocol reviewed by Dr. Bepar Mohammad Tayyib, MBBS (Medical Advisor). 
        Assistive screening aid — <a href="https://github.com/meowhuriya/tb-screen-ai#readme" target="_blank">Documentation and Clinical Benchmarks</a>
    </div>
    """,
    unsafe_allow_html=True,
)
