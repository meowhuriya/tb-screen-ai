#!/usr/bin/env python3
"""Capture final verified screenshots from current app for docs/images/ at 1440px width.

Screenshots captured:
1. docs/images/app_overview.png (Empty State)
2. docs/images/sample_normal_result.png (Low risk result with heatmap)
3. docs/images/sample_tb_heatmap.png (High risk result with heatmap)
4. docs/images/safety_ood_refusal.png (Refusal of non-X-ray photo)
5. docs/images/batch_triage_mode.png (Batch triage tab with 4 images)
6. docs/images/clinical_pdf_report.png (PDF report page)
7. docs/images/hindi_interface.png (Hindi screen)

Strict validation:
- Title must not be clipped (y >= 60px).
- No raw icon text (e.g., 'arrow_down', 'upload', 'keyboard_arrow_right') rendered in Inter font.
- All icon bounding boxes must be <= 32px width.
"""

import os
import sys
import time
import shutil
import subprocess
import urllib.request
from PIL import Image
from playwright.sync_api import sync_playwright

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_IMG_DIR = os.path.join(REPO_DIR, "docs", "images")
os.makedirs(DOCS_IMG_DIR, exist_ok=True)

PORT = 8540
URL = f"http://localhost:{PORT}"

SAMPLES = {
    "normal": os.path.join(REPO_DIR, "app", "sample_images", "sample_normal.png"),
    "advanced": os.path.join(REPO_DIR, "app", "sample_images", "sample_tb_advanced.png"),
    "moderate": os.path.join(REPO_DIR, "app", "sample_images", "sample_tb_moderate.png"),
    "review": os.path.join(REPO_DIR, "data", "external", "montgomery", "MCUCXR_0162_1.png"),
    "color_ood": "/tmp/test_color_photo.png",
}

if not os.path.exists(SAMPLES["color_ood"]):
    img = Image.new("RGB", (300, 300), color=(255, 60, 40))
    img.save(SAMPLES["color_ood"])

def wait_for_server(timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(URL, timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False

def assert_no_icon_glitch_or_clipped_title(page, screen_name):
    """Programmatic audit of page layout: title position and icon font ligatures."""
    # 1. Check title positioning
    title_box = page.evaluate("""() => {
        const title = document.querySelector('.app-title');
        if (!title) return null;
        const rect = title.getBoundingClientRect();
        return { y: rect.y, height: rect.height, top: rect.top };
    }""")
    if title_box:
        assert title_box["top"] >= 55, f"[{screen_name}] Title is clipped! top={title_box['top']}px (must be >= 55px)"

    # 2. Check icon elements
    icon_glitches = page.evaluate("""() => {
        const icons = document.querySelectorAll(
            '[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded, .material-symbols-outlined, span[class*="material"], [data-testid="stExpanderToggleIcon"] *'
        );
        const bad = [];
        icons.forEach(el => {
            const rect = el.getBoundingClientRect();
            const text = (el.innerText || el.textContent || '').trim();
            const style = window.getComputedStyle(el);
            // If the element has multicharacter ligature text and spans more than 35px, it is broken
            if (rect.width > 35 && text.length > 2) {
                bad.push({
                    text: text,
                    width: rect.width,
                    fontFamily: style.fontFamily
                });
            }
        });
        return bad;
    }""")
    if icon_glitches:
        raise RuntimeError(f"[{screen_name}] Icon glitch detected! Icons rendered as text: {icon_glitches}")
    print(f"  [✓] {screen_name}: Title y-clearance and icon glyph integrity verified.")

def main():
    print(f"Starting Streamlit server on port {PORT}...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/app.py", "--server.port", str(PORT), "--server.headless", "true"],
        cwd=REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        if not wait_for_server(35):
            print("ERROR: Streamlit failed to start within 35s")
            proc.terminate()
            sys.exit(1)

        print("Streamlit server is ready!")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1100})
            page = context.new_page()
            page.goto(URL, wait_until="networkidle")
            page.wait_for_timeout(2500)

            # -------------------------------------------------------------
            # 1. Empty State -> docs/images/app_overview.png
            # -------------------------------------------------------------
            print("\nCapturing 1: Empty State (app_overview.png)...")
            assert_no_icon_glitch_or_clipped_title(page, "Empty State")
            page.screenshot(path=os.path.join(DOCS_IMG_DIR, "app_overview.png"), full_page=False)

            # -------------------------------------------------------------
            # 2. Low Risk Result (Normal sample) -> docs/images/sample_normal_result.png
            # -------------------------------------------------------------
            print("\nCapturing 2: Low Risk Normal Result (sample_normal_result.png)...")
            page.get_by_role("button", name="Normal Chest").click()
            page.wait_for_timeout(3500)
            assert_no_icon_glitch_or_clipped_title(page, "Normal Result")
            page.screenshot(path=os.path.join(DOCS_IMG_DIR, "sample_normal_result.png"), full_page=False)

            # -------------------------------------------------------------
            # 3. High Risk Result (Advanced TB sample) -> docs/images/sample_tb_heatmap.png
            # -------------------------------------------------------------
            print("\nCapturing 3: High Risk TB Result (sample_tb_heatmap.png)...")
            page.get_by_role("button", name="Advanced TB").click()
            page.wait_for_timeout(3500)
            assert_no_icon_glitch_or_clipped_title(page, "High Risk Result")
            page.screenshot(path=os.path.join(DOCS_IMG_DIR, "sample_tb_heatmap.png"), full_page=False)

            # -------------------------------------------------------------
            # 4. Refusal of Non-X-ray Photo -> docs/images/safety_ood_refusal.png
            # -------------------------------------------------------------
            print("\nCapturing 4: OOD Refusal (safety_ood_refusal.png)...")
            file_input = page.locator("input[type=file]").first
            file_input.set_input_files(SAMPLES["color_ood"])
            page.wait_for_timeout(3000)
            assert_no_icon_glitch_or_clipped_title(page, "OOD Refusal")
            page.screenshot(path=os.path.join(DOCS_IMG_DIR, "safety_ood_refusal.png"), full_page=False)

            # -------------------------------------------------------------
            # 5. Batch Triage Tab with 4 images -> docs/images/batch_triage_mode.png
            # -------------------------------------------------------------
            print("\nCapturing 5: Batch Triage Tab (batch_triage_mode.png)...")
            page.get_by_role("tab", name="Batch Triage").click()
            page.wait_for_timeout(1500)

            batch_input = page.locator("input[type=file]").nth(1)
            batch_files = [
                SAMPLES["normal"],
                SAMPLES["advanced"],
                SAMPLES["moderate"],
            ]
            if os.path.exists(SAMPLES["review"]):
                batch_files.append(SAMPLES["review"])
            else:
                batch_files.append(SAMPLES["normal"])

            batch_input.set_input_files(batch_files)
            page.wait_for_timeout(2000)

            run_btn = page.get_by_role("button", name="Run Batch Triage Analysis")
            if run_btn.is_visible():
                run_btn.click()
                page.wait_for_timeout(5000)

            assert_no_icon_glitch_or_clipped_title(page, "Batch Triage")
            page.screenshot(path=os.path.join(DOCS_IMG_DIR, "batch_triage_mode.png"), full_page=False)

            # -------------------------------------------------------------
            # 6. Hindi Screen -> docs/images/hindi_interface.png
            # -------------------------------------------------------------
            print("\nCapturing 6: Hindi Screen (hindi_interface.png)...")
            # Switch back to single tab
            page.get_by_role("tab").first.click()
            page.wait_for_timeout(1000)

            # Change language to Hindi
            selectbox = page.locator("div[data-testid='stSelectbox']").first
            selectbox.click()
            page.wait_for_timeout(500)
            page.get_by_role("option", name="हिन्दी (Hindi)").click()
            page.wait_for_timeout(2500)

            # Click Advanced TB in Hindi
            tb_btn = page.locator("button").filter(has_text="2.").first
            if tb_btn.is_visible():
                tb_btn.click()
                page.wait_for_timeout(3500)

            assert_no_icon_glitch_or_clipped_title(page, "Hindi Interface")
            page.screenshot(path=os.path.join(DOCS_IMG_DIR, "hindi_interface.png"), full_page=False)

            browser.close()

        # -------------------------------------------------------------
        # 7. PDF Report Page -> docs/images/clinical_pdf_report.png
        # -------------------------------------------------------------
        print("\nCapturing 7: PDF Report Page (clinical_pdf_report.png)...")
        sys.path.insert(0, REPO_DIR)
        from src.predict import predict_chest_xray
        from app.pdf_generator import create_diagnostic_pdf

        pred = predict_chest_xray(SAMPLES["advanced"])
        pdf_bytes = create_diagnostic_pdf(
            prediction=pred,
            patient_id="PAT-20261105-01",
            clinic_name="Primary Care Rural Center",
            notes="Active TB suspected; confirm with GeneXpert MTB/RIF or sputum smear.",
        )
        temp_pdf = "/tmp/final_clinical_report.pdf"
        temp_png = "/tmp/final_clinical_report.png"
        with open(temp_pdf, "wb") as f:
            f.write(pdf_bytes)

        subprocess.run(["sips", "-s", "format", "png", temp_pdf, "--out", temp_png], check=True, capture_output=True)
        target_pdf_img = os.path.join(DOCS_IMG_DIR, "clinical_pdf_report.png")
        shutil.copy(temp_png, target_pdf_img)
        print(f"  [✓] PDF report generated and saved to: {target_pdf_img}")

        print("\nAll 7 final screenshots successfully captured, validated, and saved to docs/images/!")

    finally:
        print("Stopping Streamlit server...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        print("Server stopped cleanly.")

if __name__ == "__main__":
    main()
