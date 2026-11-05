"""Clinical Hospital Design System & CSS Injections for TB-Screen AI.

Hospital Grade UI:
- Primary Color: Teal #0E6E8C
- Secondary / Neutral: Slate #1B2733, Muted #5B6B7B, Border #E3E8EE
- Backgrounds: Canvas #F5F7FA, Surface #FFFFFF
- Risk Colors (used only for risk states):
  * Low Risk: Text #1E7B4F, Background #E8F5EE, Border #C6E7D2
  * Needs Review: Text #9A6700, Background #FFF4DB, Border #FFE6A3
  * High Risk: Text #B42318, Background #FDECEA, Border #F9D2CE
- Font: Inter (Google Fonts) with system-sans fallback
- 8px baseline grid, 12px card border radius, soft 0 1px 2px shadow
- Max width 1100px centered layout
- Zero emojis; clean Material icons
"""

import streamlit as st


def inject_custom_css() -> None:
    """Inject hospital-grade clinical styling into Streamlit DOM."""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* Selective Typography: Inter applied to text, headings, buttons, inputs, tabs */
    html, body, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #1B2733;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    p, .stMarkdown p, .stText, h1, h2, h3, h4, h5, h6,
    input, textarea, select, label,
    button:not([data-testid*="Collapse"]):not([data-testid*="header"]),
    .stButton > button,
    [data-testid="stTabs"] button,
    .clinical-card, .patient-strip, .next-step-banner {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #1B2733;
    }

    /* Explicitly preserve Streamlit Material Icon Fonts */
    [data-testid="stIconMaterial"],
    .material-icons,
    .material-symbols-rounded,
    .material-symbols-outlined,
    span[class*="material"],
    i[class*="material"],
    [data-testid="stExpanderToggleIcon"] *,
    [data-testid*="CollapseButton"] *,
    [data-testid*="headerNoPadding"] *,
    [data-testid="stFileUploaderDropzone"] span[data-testid="stIconMaterial"],
    [data-testid="stSelectbox"] span[data-testid="stIconMaterial"],
    summary span:first-child span {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
    }

    /* Constrain main app container to 1100px centered with enough top clearance below Streamlit header */
    .block-container {
        max-width: 1100px !important;
        padding-top: 4.25rem !important;
        padding-bottom: 3.0rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        margin: 0 auto !important;
    }

    /* Overall background */
    .stApp {
        background-color: #F5F7FA !important;
    }

    /* Headers */
    .app-title {
        font-size: 28px !important;
        font-weight: 600 !important;
        color: #1B2733 !important;
        letter-spacing: -0.02em !important;
        margin: 0 0 4px 0 !important;
        line-height: 1.25 !important;
    }

    .app-subtitle {
        font-size: 15px !important;
        color: #5B6B7B !important;
        margin: 0 0 16px 0 !important;
        font-weight: 400 !important;
        line-height: 1.4 !important;
    }

    .section-title {
        font-size: 18px !important;
        font-weight: 600 !important;
        color: #1B2733 !important;
        margin: 0 0 12px 0 !important;
        letter-spacing: -0.01em !important;
    }

    .caption-muted {
        font-size: 13px !important;
        color: #5B6B7B !important;
        line-height: 1.4 !important;
    }

    /* Header Pill Badge: Screening aid - not a diagnosis */
    .header-aid-badge {
        display: inline-block;
        padding: 5px 12px;
        background: #FFFFFF;
        border: 1px solid #E3E8EE;
        border-radius: 20px;
        color: #5B6B7B;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    /* Clinical Cards */
    .clinical-card {
        background: #FFFFFF;
        border: 1px solid #E3E8EE;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
        padding: 24px;
        margin-bottom: 20px;
    }

    .patient-strip {
        background: #FFFFFF;
        border: 1px solid #E3E8EE;
        border-radius: 8px;
        padding: 12px 18px;
        margin-bottom: 16px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        font-size: 13.5px;
        color: #1B2733;
    }

    .patient-strip-item {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }

    .patient-strip-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #5B6B7B;
        font-weight: 600;
    }

    .patient-strip-val {
        font-weight: 600;
        color: #1B2733;
    }

    /* Result Card Elements */
    .result-status-badge {
        display: inline-flex;
        align-items: flex-start;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.01em;
        line-height: 1.4;
        max-width: 100%;
        word-break: break-word;
        margin-bottom: 12px;
    }

    .result-status-badge svg {
        flex-shrink: 0;
        margin-top: 2px;
    }

    .badge-low-risk {
        background-color: #E8F5EE;
        color: #1E7B4F;
        border: 1px solid #C6E7D2;
    }

    .badge-review {
        background-color: #FFF4DB;
        color: #9A6700;
        border: 1px solid #FFE6A3;
    }

    .badge-high-risk {
        background-color: #FDECEA;
        color: #B42318;
        border: 1px solid #F9D2CE;
    }

    .probability-display {
        font-size: 40px;
        font-weight: 700;
        color: #1B2733;
        line-height: 1.1;
        letter-spacing: -0.02em;
        margin: 4px 0 10px 0;
    }

    .probability-label {
        font-size: 13px;
        color: #5B6B7B;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    /* Progress bar with active threshold marker */
    .prob-track {
        width: 100%;
        height: 8px;
        background: #E3E8EE;
        border-radius: 4px;
        position: relative;
        margin: 10px 0 16px 0;
        overflow: visible;
    }

    .prob-fill-low {
        height: 100%;
        border-radius: 4px;
        background: #1E7B4F;
    }

    .prob-fill-review {
        height: 100%;
        border-radius: 4px;
        background: #9A6700;
    }

    .prob-fill-high {
        height: 100%;
        border-radius: 4px;
        background: #B42318;
    }

    .threshold-tick {
        position: absolute;
        top: -4px;
        width: 2px;
        height: 16px;
        background: #1B2733;
        border-radius: 1px;
    }

    .clinical-impression-text {
        font-size: 15px;
        color: #1B2733;
        line-height: 1.45;
        margin: 8px 0 12px 0;
    }

    .next-step-banner {
        background: #F8FAFC;
        border-left: 3px solid #0E6E8C;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        font-size: 14px;
        line-height: 1.4;
        color: #1B2733;
        margin-top: 12px;
    }

    /* Radiology Viewer Panels */
    .radiology-box {
        background-color: #0B0F14;
        border: 1px solid #1E293B;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        max-width: 380px;
        margin: 0 auto;
    }

    .radiology-caption {
        font-size: 12px;
        color: #94A3B8;
        font-weight: 500;
        margin-bottom: 8px;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }

    .legend-bar {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 3px;
        margin-top: 8px;
        max-width: 380px;
        margin-left: auto;
        margin-right: auto;
    }

    .legend-gradient {
        width: 100%;
        height: 4px;
        border-radius: 2px;
        background: linear-gradient(to right, #0000FF, #00FFFF, #00FF00, #FFFF00, #FF0000);
    }

    .legend-labels {
        width: 100%;
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        color: #5B6B7B;
    }

    /* Buttons: Minimum 44px height, accessible focus */
    div.stButton > button {
        min-height: 44px !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 14.5px !important;
        border: 1px solid #E3E8EE !important;
        background-color: #FFFFFF !important;
        color: #1B2733 !important;
        transition: all 0.15s ease-in-out !important;
        padding: 8px 16px !important;
    }

    div.stButton > button:hover {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
    }

    div.stButton > button:focus-visible {
        outline: 2px solid #0E6E8C !important;
        outline-offset: 2px !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: #0E6E8C !important;
        color: #FFFFFF !important;
        border: none !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #0B5870 !important;
    }

    /* Streamlit Tabs Styling: clean line underline, plain text */
    button[data-baseweb="tab"] {
        font-size: 15px !important;
        font-weight: 500 !important;
        color: #5B6B7B !important;
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 10px 18px !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #1B2733 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0E6E8C !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #0E6E8C !important;
    }

    /* Clean expander styling */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E3E8EE !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        margin-bottom: 12px !important;
    }

    /* Small clean status tag */
    .scanner-tag-familiar {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        background: #E8F5EE;
        color: #1E7B4F;
        border: 1px solid #C6E7D2;
    }

    .scanner-tag-unfamiliar {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        background: #FFF4DB;
        color: #9A6700;
        border: 1px solid #FFE6A3;
    }

    /* Footer styling */
    .clinic-footer {
        margin-top: 40px;
        padding-top: 16px;
        border-top: 1px solid #E3E8EE;
        font-size: 12.5px;
        color: #5B6B7B;
        text-align: center;
    }

    .clinic-footer a {
        color: #0E6E8C;
        text-decoration: none;
    }

    .clinic-footer a:hover {
        text-decoration: underline;
    }

    /* Responsive adjustments */
    @media (max-width: 700px) {
        .block-container {
            padding-top: 4.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .patient-strip {
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }
        .probability-display {
            font-size: 32px;
        }
        .radiology-box {
            max-width: 100%;
        }
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
