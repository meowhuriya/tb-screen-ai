#!/usr/bin/env python3
"""Capture high-resolution real screenshots and animated demo GIF from the live Streamlit app.

Generates:
- docs/images/app_overview.png
- docs/images/sample_tb_heatmap.png
- docs/images/sample_normal_result.png
- docs/images/safety_ood_refusal.png
- docs/images/clinical_pdf_report.png
- docs/images/batch_triage_mode.png
- docs/images/scanner_shift_badge.png
- docs/images/demo.gif
"""

import os
import sys
import subprocess
import time
from PIL import Image

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from playwright.sync_api import sync_playwright
DOCS_IMG_DIR = os.path.join(REPO_DIR, "docs", "images")
os.makedirs(DOCS_IMG_DIR, exist_ok=True)

# 1. Generate Clinical PDF Report image first
print("[1/8] Generating Clinical Referral PDF Report Image...")
from src.predict import predict_chest_xray
from app.pdf_generator import create_diagnostic_pdf

tb_sample_path = os.path.join(REPO_DIR, "app", "sample_images", "sample_tb_advanced.png")
tb_pil = Image.open(tb_sample_path)
tb_pred = predict_chest_xray(tb_pil, threshold=0.05)
pdf_bytes = create_diagnostic_pdf(
    prediction=tb_pred,
    patient_id="PAT-20261105-01",
    clinic_name="Primary Health Centre (Utnoor Rural Screening)",
    notes="Persistent chronic cough > 3 weeks, evening pyrexia, hemoptysis. Priority GeneXpert recommended."
)

temp_pdf_path = "/tmp/tb_screening_referral_sample.pdf"
with open(temp_pdf_path, "wb") as f:
    f.write(pdf_bytes)

# Convert PDF to high-res image using macOS QuickLook
subprocess.run(["qlmanage", "-t", "-s", "1200", "-o", "/tmp", temp_pdf_path], capture_output=True)
pdf_img_path = temp_pdf_path + ".png"
if os.path.exists(pdf_img_path):
    pdf_img = Image.open(pdf_img_path)
    pdf_img.save(os.path.join(DOCS_IMG_DIR, "clinical_pdf_report.png"))
    print("  -> Saved docs/images/clinical_pdf_report.png")
else:
    print("  -> Warning: QuickLook thumbnail not generated, falling back.")

# 2. Launch Playwright for web UI screenshots
print("[2/8] Launching headless browser against http://localhost:8501...")
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    
    # Common viewport
    page = browser.new_page(viewport={"width": 1400, "height": 1150})
    page.goto("http://localhost:8501", wait_until="networkidle")
    page.wait_for_timeout(2500)

    # -------------------------------------------------------------
    # Screenshot 1: TB Sample with Heatmap & App Overview
    # -------------------------------------------------------------
    print("[3/8] Capturing TB Sample with Grad-CAM and Shift Badge...")
    page.get_by_text("2. Advanced TB (Cavitation)").click()
    page.wait_for_timeout(3500)
    
    # Save app overview and TB sample heatmap
    app_overview_path = os.path.join(DOCS_IMG_DIR, "app_overview.png")
    tb_heatmap_path = os.path.join(DOCS_IMG_DIR, "sample_tb_heatmap.png")
    page.screenshot(path=app_overview_path, full_page=False)
    page.screenshot(path=tb_heatmap_path, full_page=False)
    print("  -> Saved docs/images/app_overview.png & sample_tb_heatmap.png")

    # -------------------------------------------------------------
    # Screenshot 2: Normal Sample
    # -------------------------------------------------------------
    print("[4/8] Capturing Normal Healthy Chest Result...")
    page.get_by_text("1. Normal Chest (Healthy)").click()
    page.wait_for_timeout(3500)
    normal_path = os.path.join(DOCS_IMG_DIR, "sample_normal_result.png")
    page.screenshot(path=normal_path, full_page=False)
    print("  -> Saved docs/images/sample_normal_result.png")

    # -------------------------------------------------------------
    # Screenshot 3: Refusal of Non-X-ray Photo (OOD Defense)
    # -------------------------------------------------------------
    print("[5/8] Capturing Out-of-Distribution Rejection...")
    page.get_by_text("None (Upload Custom)").click()
    page.wait_for_timeout(1000)
    
    # Upload colorful non-X-ray
    file_uploader = page.locator("input[type=file]").first
    file_uploader.set_input_files("/tmp/test_color_photo.png")
    page.wait_for_timeout(3000)
    ood_path = os.path.join(DOCS_IMG_DIR, "safety_ood_refusal.png")
    page.screenshot(path=ood_path, full_page=False)
    print("  -> Saved docs/images/safety_ood_refusal.png")

    # -------------------------------------------------------------
    # Screenshot 4: Scanner-Shift Domain Badge (Unfamiliar scan)
    # -------------------------------------------------------------
    print("[6/8] Capturing Scanner-Shift Domain Badges...")
    # Find an unfamiliar Montgomery image
    mont_unfamiliar_file = os.path.join(REPO_DIR, "data/external/montgomery/MCUCXR_0023_0.png")
    if os.path.exists(mont_unfamiliar_file):
        file_uploader.set_input_files(mont_unfamiliar_file)
        page.wait_for_timeout(3500)
        shift_path = os.path.join(DOCS_IMG_DIR, "scanner_shift_badge.png")
        page.screenshot(path=shift_path, full_page=False)
        print("  -> Saved docs/images/scanner_shift_badge.png (showing Unfamiliar source warning)")

    # -------------------------------------------------------------
    # Screenshot 5: Batch Triage Mode
    # -------------------------------------------------------------
    print("[7/8] Capturing High-Volume Batch Triage Mode...")
    page.get_by_text("High-Volume Batch Triage").click()
    page.wait_for_timeout(1500)

    batch_uploader = page.locator("input[type=file]").nth(1)
    batch_samples = [
        os.path.join(REPO_DIR, "app/sample_images/sample_tb_advanced.png"),
        os.path.join(REPO_DIR, "app/sample_images/sample_tb_moderate.png"),
        os.path.join(REPO_DIR, "app/sample_images/sample_normal.png")
    ]
    batch_uploader.set_input_files(batch_samples)
    page.wait_for_timeout(2000)

    run_btn = page.get_by_role("button", name="Run Batch Triage Analysis")
    if run_btn.is_visible():
        run_btn.click()
        page.wait_for_timeout(4500)

    batch_path = os.path.join(DOCS_IMG_DIR, "batch_triage_mode.png")
    page.screenshot(path=batch_path, full_page=False)
    print("  -> Saved docs/images/batch_triage_mode.png")

    browser.close()

# -------------------------------------------------------------
# Step 8: Build Animated Demo GIF
# -------------------------------------------------------------
print("[8/8] Assembling animated demo GIF...")
gif_frames = []
frame_paths = [
    os.path.join(DOCS_IMG_DIR, "app_overview.png"),
    os.path.join(DOCS_IMG_DIR, "sample_normal_result.png"),
    os.path.join(DOCS_IMG_DIR, "safety_ood_refusal.png"),
    os.path.join(DOCS_IMG_DIR, "batch_triage_mode.png"),
    os.path.join(DOCS_IMG_DIR, "clinical_pdf_report.png"),
]

for fp in frame_paths:
    if os.path.exists(fp):
        im = Image.open(fp).convert("RGB")
        # Resize uniformly to 1000px width for compact fast-loading web GIF
        w, h = im.size
        new_w = 900
        new_h = int(h * (new_w / w))
        im_resized = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        gif_frames.append(im_resized)

if gif_frames:
    demo_gif_path = os.path.join(DOCS_IMG_DIR, "demo.gif")
    gif_frames[0].save(
        demo_gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=2200,  # 2.2 seconds per frame
        loop=0
    )
    print(f"  -> Generated docs/images/demo.gif ({os.path.getsize(demo_gif_path) / 1024:.1f} KB)")

print("\nAll real screenshots and demo GIF generated successfully!")
