"""PDF Referral Report Generator for TB-Screen AI.

Generates a clean, professional, single-page A4 clinical diagnostic summary:
- Patient & clinic metadata
- Original chest X-ray side-by-side with Grad-CAM heatmap overlay
- Traffic-light risk badge & calibrated confidence score
- Step-by-step clinical triage action plan
- Sign-off box for local Medical Officer
"""

import datetime
import os
import tempfile
from typing import Any, Dict

from fpdf import FPDF
from PIL import Image


class ClinicalPDF(FPDF):
    """Custom formatted clinical report PDF with hospital-grade palette."""

    def header(self):
        # Header banner in primary blue-teal #0E6E8C
        self.set_fill_color(14, 110, 140)
        self.rect(0, 0, 210, 22, "F")
        self.set_font("helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.set_xy(12, 6)
        self.cell(0, 8, "TB-SCREEN AI  |  Frontline Pulmonary Diagnostic Summary", new_x="RIGHT", new_y="TOP", align="L")
        self.set_font("helvetica", "", 8.5)
        self.set_xy(12, 13.5)
        self.cell(0, 6, "Rural Health Triage & Primary Care Screening System (HI-01)", new_x="RIGHT", new_y="TOP", align="L")

    def footer(self):
        self.set_y(-18)
        self.set_font("helvetica", "", 7.5)
        self.set_text_color(91, 107, 123)  # #5B6B7B
        disclaimer = (
            "REGULATORY NOTICE: Frontline screening assistance only. Not an autonomous diagnostic device. "
            "Workflow reviewed by Dr. Bepar Mohammad Tayyib, MBBS (Medical Advisor). "
            "Corroborate with GeneXpert / Sputum Smear. National TB Helpline: 1800-11-6666 | nikshay.in"
        )
        self.multi_cell(0, 4, disclaimer, align="C")
        self.set_y(-8)
        self.cell(0, 5, f"Page {self.page_no()}", new_x="RIGHT", new_y="TOP", align="R")


def create_diagnostic_pdf(
    prediction: Dict[str, Any],
    patient_id: str = "PT-UNKNOWN",
    patient_name: str = "Anonymous Patient",
    clinic_name: str = "Primary Health Center (PHC)",
    notes: str = "",
) -> bytes:
    """Generate single-page diagnostic PDF report.

    Args:
        prediction: Dictionary from predict_chest_xray.
        patient_id: Identifier for patient.
        patient_name: Name of patient.
        clinic_name: Clinic facility name.
        notes: Free-form clinical notes.

    Returns:
        bytes: PDF binary content.
    """
    pdf = ClinicalPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # Metadata Strip (White card with #E3E8EE border)
    pdf.set_y(26)
    pdf.set_fill_color(255, 255, 255)
    pdf.rect(10, 26, 190, 22, "F")
    pdf.set_draw_color(227, 232, 238)  # #E3E8EE
    pdf.rect(10, 26, 190, 22, "D")

    now_str = datetime.datetime.now().strftime("%d-%b-%Y %H:%M")
    pdf.set_text_color(27, 39, 51)  # #1B2733

    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_xy(14, 29)
    pdf.cell(28, 5, "Patient ID:", new_x="RIGHT", new_y="TOP")
    pdf.set_font("helvetica", "", 8.5)
    pdf.cell(55, 5, str(patient_id), new_x="RIGHT", new_y="TOP")

    pdf.set_font("helvetica", "B", 8.5)
    pdf.cell(26, 5, "Date & Time:", new_x="RIGHT", new_y="TOP")
    pdf.set_font("helvetica", "", 8.5)
    pdf.cell(50, 5, now_str, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_xy(14, 37)
    pdf.cell(28, 5, "Facility / Clinic:", new_x="RIGHT", new_y="TOP")
    pdf.set_font("helvetica", "", 8.5)
    pdf.cell(55, 5, str(clinic_name), new_x="RIGHT", new_y="TOP")

    pdf.set_font("helvetica", "B", 8.5)
    pdf.cell(26, 5, "Model Engine:", new_x="RIGHT", new_y="TOP")
    pdf.set_font("helvetica", "", 8.5)
    pdf.cell(50, 5, "EfficientNet-B0 (Edge PyTorch)", new_x="LMARGIN", new_y="NEXT")

    # Temporary image exports for embedding
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_img_path = os.path.join(tmpdir, "orig.jpg")
        cam_img_path = os.path.join(tmpdir, "cam.jpg")

        orig_pil = prediction["original_image_pil"].resize((350, 350))
        orig_pil.save(orig_img_path, format="JPEG", quality=92)

        cam_overlay = prediction["gradcam_overlay_rgb"]
        cam_pil = Image.fromarray(cam_overlay).resize((350, 350))
        cam_pil.save(cam_img_path, format="JPEG", quality=92)

        # Place images side-by-side with dark viewer frame
        pdf.set_y(52)
        pdf.set_fill_color(11, 15, 20)  # #0B0F14 dark radiology panel
        pdf.rect(14, 52, 87, 87, "F")
        pdf.rect(109, 52, 87, 87, "F")
        pdf.image(orig_img_path, x=15, y=53, w=85, h=85)
        pdf.image(cam_img_path, x=110, y=53, w=85, h=85)

        # Image Subtitles
        pdf.set_font("helvetica", "B", 8.5)
        pdf.set_text_color(91, 107, 123)
        pdf.set_xy(14, 140)
        pdf.cell(87, 5, "Original Frontal Chest Radiograph", new_x="RIGHT", new_y="TOP", align="C")
        pdf.set_xy(109, 140)
        pdf.cell(87, 5, "Grad-CAM Pulmonary Attention Overlay", new_x="LMARGIN", new_y="NEXT", align="C")

    # Risk Assessment Box with hospital palette
    risk_tier = prediction["risk_tier"]
    tb_prob = prediction["tb_probability"]
    conf = prediction["confidence_percentage"]

    if risk_tier == "HIGH":
        fill_rgb = (253, 236, 234)  # #FDECEA
        border_rgb = (249, 210, 206)
        text_rgb = (180, 35, 24)    # #B42318
        risk_label = "High risk. TB suspected; confirm with a sputum test (GeneXpert or smear)."
        guidance = "Provide surgical mask. Fast-track urgent sputum GeneXpert/microscopy and refer to DOTS clinic. National TB Helpline: 1800-11-6666 | nikshay.in"
    elif risk_tier == "NEEDS_REVIEW":
        fill_rgb = (255, 244, 219)  # #FFF4DB
        border_rgb = (255, 230, 163)
        text_rgb = (154, 103, 0)    # #9A6700
        risk_label = "Needs review. Findings are indeterminate; clinician review advised."
        guidance = "Findings are indeterminate. Review by Medical Officer advised. Collect sputum for acid-fast smear or GeneXpert MTB/RIF assay. National TB Helpline: 1800-11-6666"
    else:
        fill_rgb = (232, 245, 238)  # #E8F5EE
        border_rgb = (198, 231, 210)
        text_rgb = (30, 123, 79)    # #1E7B4F
        risk_label = "Low risk. No TB pattern detected by the model."
        guidance = "No TB pattern detected by the model. Routine primary care. Advise patient to return if persistent cough (>2 weeks), fever, or night sweats develop."

    pdf.set_y(147)
    pdf.set_fill_color(fill_rgb[0], fill_rgb[1], fill_rgb[2])
    pdf.set_draw_color(border_rgb[0], border_rgb[1], border_rgb[2])
    pdf.rect(10, 147, 190, 10, "FD")
    pdf.set_font("helvetica", "B", 9.5)
    pdf.set_text_color(text_rgb[0], text_rgb[1], text_rgb[2])
    pdf.set_xy(10, 148.5)
    pdf.cell(190, 7, risk_label, new_x="LMARGIN", new_y="NEXT", align="C")

    # Quantitative Metrics Row
    pdf.set_y(160)
    pdf.set_text_color(27, 39, 51)
    pdf.set_font("helvetica", "B", 9)
    pdf.cell(48, 6, f"TB Probability: {tb_prob*100:.1f}%", new_x="RIGHT", new_y="TOP", align="L")
    pdf.cell(48, 6, f"Confidence Score: {conf:.1f}%", new_x="RIGHT", new_y="TOP", align="L")
    pdf.cell(48, 6, "Calibrated Cutoff: 27.0%", new_x="RIGHT", new_y="TOP", align="L")
    pdf.cell(46, 6, f"Diagnostic Status: {prediction['predicted_label']}", new_x="LMARGIN", new_y="NEXT", align="L")

    # Action Guidance Box
    pdf.set_y(169)
    pdf.set_fill_color(248, 250, 252)  # #F8FAFC
    pdf.rect(10, 169, 190, 24, "F")
    pdf.set_draw_color(227, 232, 238)  # #E3E8EE
    pdf.rect(10, 169, 190, 24, "D")

    # Teal left highlight line
    pdf.set_fill_color(14, 110, 140)
    pdf.rect(10, 169, 2.5, 24, "F")

    pdf.set_xy(15, 171)
    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_text_color(14, 110, 140)
    pdf.cell(0, 5, "Recommended Clinical Action Protocol:", new_x="LMARGIN", new_y="NEXT")

    pdf.set_xy(15, 176)
    pdf.set_font("helvetica", "", 8.5)
    pdf.set_text_color(27, 39, 51)
    pdf.multi_cell(180, 4.5, guidance)

    # Clinical Notes & Doctor Signoff
    pdf.set_y(196)
    pdf.set_draw_color(227, 232, 238)
    pdf.rect(10, 196, 120, 36, "D")
    pdf.set_xy(12, 198)
    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_text_color(91, 107, 123)
    pdf.cell(0, 4, "Field Observations & Notes:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(12, 203)
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(91, 107, 123)
    note_text = notes if notes else "No additional notes recorded by field health worker."
    pdf.multi_cell(116, 4, note_text)

    # Doctor Signoff Box
    pdf.rect(135, 196, 65, 36, "D")
    pdf.set_xy(137, 198)
    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_text_color(27, 39, 51)
    pdf.cell(60, 4, "Medical Officer Sign-off", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.line(140, 222, 195, 222)
    pdf.set_xy(137, 223)
    pdf.set_font("helvetica", "I", 7.5)
    pdf.set_text_color(91, 107, 123)
    pdf.cell(60, 4, "Signature & Date / Official Stamp", new_x="LMARGIN", new_y="NEXT", align="C")

    # Return raw PDF bytes
    return bytes(pdf.output())
