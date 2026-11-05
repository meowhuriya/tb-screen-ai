"""PDF Referral Report Generator for TB-Screen AI.

Generates a clean, professional, single-page A4 clinical diagnostic summary:
- Patient & clinic metadata
- Original chest X-ray side-by-side with Grad-CAM heatmap overlay
- Traffic light risk badge & confidence score
- Step-by-step clinical triage action plan
- Sign-off box for local Medical Officer
"""

import datetime
import io
import os
import tempfile
from typing import Any, Dict

import numpy as np
from fpdf import FPDF
from PIL import Image


class ClinicalPDF(FPDF):
    """Custom formatted clinical report PDF."""

    def header(self):
        # Header banner
        self.set_fill_color(24, 43, 73)  # Clinical navy blue
        self.rect(0, 0, 210, 22, "F")
        self.set_font("helvetica", "B", 14)
        self.set_text_color(255, 255, 255)
        self.set_xy(10, 6)
        self.cell(0, 10, "TB-SCREEN AI  |  Frontline Pulmonary Diagnostic Summary", new_x="RIGHT", new_y="TOP", align="L")
        self.set_font("helvetica", "I", 9)
        self.set_xy(10, 14)
        self.cell(0, 6, "Rural Health Triage & Primary Care Screening System (HI-01)", new_x="RIGHT", new_y="TOP", align="L")

    def footer(self):
        self.set_y(-18)
        self.set_font("helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        disclaimer = (
            "REGULATORY NOTICE: Frontline screening assistance only. Not an autonomous diagnostic device. "
            "Workflow reviewed by Dr. Bepar Mohammad Tayyib, MBBS (Clinical Advisor). "
            "Corroborate with GeneXpert/Sputum Smear. National TB Helpline: 1800-11-6666 | nikshay.in"
        )
        self.multi_cell(0, 4, disclaimer, align="C")
        self.set_y(-8)
        self.cell(0, 6, f"Page {self.page_no()}", new_x="RIGHT", new_y="TOP", align="R")


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

    # Metadata Card
    pdf.set_y(26)
    pdf.set_fill_color(245, 247, 250)
    pdf.rect(10, 26, 190, 22, "F")
    pdf.set_draw_color(210, 215, 225)
    pdf.rect(10, 26, 190, 22, "D")

    now_str = datetime.datetime.now().strftime("%d-%b-%Y %H:%M")
    pdf.set_text_color(30, 30, 30)

    pdf.set_font("helvetica", "B", 9)
    pdf.set_xy(14, 28)
    pdf.cell(32, 5, "Patient ID:", new_x="RIGHT", new_y="TOP")
    pdf.set_font("helvetica", "", 9)
    pdf.cell(50, 5, str(patient_id), new_x="RIGHT", new_y="TOP")

    pdf.set_font("helvetica", "B", 9)
    pdf.cell(28, 5, "Date & Time:", new_x="RIGHT", new_y="TOP")
    pdf.set_font("helvetica", "", 9)
    pdf.cell(50, 5, now_str, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("helvetica", "B", 9)
    pdf.set_xy(14, 36)
    pdf.cell(32, 5, "Facility / Clinic:", new_x="RIGHT", new_y="TOP")
    pdf.set_font("helvetica", "", 9)
    pdf.cell(50, 5, str(clinic_name), new_x="RIGHT", new_y="TOP")

    pdf.set_font("helvetica", "B", 9)
    pdf.cell(28, 5, "Model Engine:", new_x="RIGHT", new_y="TOP")
    pdf.set_font("helvetica", "", 9)
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

        # Place images side-by-side
        pdf.set_y(52)
        pdf.image(orig_img_path, x=15, y=52, w=85, h=85)
        pdf.image(cam_img_path, x=110, y=52, w=85, h=85)

        # Image Subtitles
        pdf.set_font("helvetica", "B", 8.5)
        pdf.set_text_color(60, 60, 60)
        pdf.set_xy(15, 138)
        pdf.cell(85, 5, "Original Frontal Chest Radiograph", new_x="RIGHT", new_y="TOP", align="C")
        pdf.set_xy(110, 138)
        pdf.cell(85, 5, "Grad-CAM Pulmonary Saliency Heatmap", new_x="LMARGIN", new_y="NEXT", align="C")

    # Risk Assessment Box
    risk_tier = prediction["risk_tier"]
    tb_prob = prediction["tb_probability"]
    conf = prediction["confidence_percentage"]

    if risk_tier == "HIGH":
        box_color = (220, 53, 69)
        risk_label = "HIGH RISK - Active Tuberculosis Indicated"
        guidance = "Immediate isolation precautions. Order urgent sputum GeneXpert/microscopy. Refer patient to District DOTS TB Center. National TB Helpline: 1800-11-6666 | nikshay.in"
    elif risk_tier == "NEEDS_REVIEW":
        box_color = (255, 193, 7)
        risk_label = "NEEDS REVIEW - Indeterminate Radiographic Finding"
        guidance = "Borderline opacities detected. Clinical correlation required by Medical Officer. Collect sputum for acid-fast smear / GeneXpert. National TB Helpline: 1800-11-6666"
    else:
        box_color = (40, 167, 69)
        risk_label = "LOW RISK - Normal Pulmonary Radiograph"
        guidance = "No radiographic patterns of active tuberculosis detected. Routine health monitoring. Re-test if cough > 2 weeks."

    pdf.set_y(146)
    pdf.set_fill_color(box_color[0], box_color[1], box_color[2])
    pdf.rect(10, 146, 190, 10, "F")
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(255, 255, 255 if risk_tier != "NEEDS_REVIEW" else 20)
    pdf.set_xy(10, 147.5)
    pdf.cell(190, 7, risk_label, new_x="LMARGIN", new_y="NEXT", align="C")

    # Quantitative Metrics Row
    pdf.set_y(159)
    pdf.set_text_color(40, 40, 40)
    pdf.set_font("helvetica", "B", 9)
    pdf.cell(48, 6, f"TB Probability: {tb_prob*100:.1f}%", new_x="RIGHT", new_y="TOP", align="L")
    pdf.cell(48, 6, f"Confidence Score: {conf:.1f}%", new_x="RIGHT", new_y="TOP", align="L")
    pdf.cell(48, 6, "Calibrated Cutoff: 27.0%", new_x="RIGHT", new_y="TOP", align="L")
    pdf.cell(46, 6, f"Diagnostic Status: {prediction['predicted_label']}", new_x="LMARGIN", new_y="NEXT", align="L")

    # Action Guidance Box
    pdf.set_y(168)
    pdf.set_fill_color(248, 249, 250)
    pdf.rect(10, 168, 190, 24, "F")
    pdf.set_draw_color(220, 224, 230)
    pdf.rect(10, 168, 190, 24, "D")

    pdf.set_xy(13, 170)
    pdf.set_font("helvetica", "B", 9)
    pdf.set_text_color(24, 43, 73)
    pdf.cell(0, 5, "Recommended Clinical Action Protocol:", new_x="LMARGIN", new_y="NEXT")

    pdf.set_xy(13, 175)
    pdf.set_font("helvetica", "", 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(184, 4.5, guidance)

    # Clinical Notes & Doctor Signoff
    pdf.set_y(195)
    pdf.set_draw_color(200, 205, 215)
    pdf.rect(10, 195, 120, 38, "D")
    pdf.set_xy(12, 197)
    pdf.set_font("helvetica", "B", 8.5)
    pdf.cell(0, 4, "Field Notes / Observations:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_xy(12, 202)
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(80, 80, 80)
    note_text = notes if notes else "No additional notes recorded by field health worker."
    pdf.multi_cell(116, 4, note_text)

    # Doctor Signoff Box
    pdf.rect(135, 195, 65, 38, "D")
    pdf.set_xy(137, 197)
    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(60, 4, "Medical Officer Sign-off", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.line(140, 222, 195, 222)
    pdf.set_xy(137, 223)
    pdf.set_font("helvetica", "I", 7.5)
    pdf.cell(60, 4, "Signature & Date / Stamp", new_x="LMARGIN", new_y="NEXT", align="C")

    # Return raw PDF bytes
    return bytes(pdf.output())
