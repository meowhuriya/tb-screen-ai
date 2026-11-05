#!/usr/bin/env python3
"""Capture hospital-grade redesign screenshots on desktop (1440px) and mobile (390px),
and refresh docs/images/ for the README.
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
SCREENSHOT_DIR = os.path.join(REPO_DIR, "docs", "images", "redesign")
MAIN_DOCS_IMG = os.path.join(REPO_DIR, "docs", "images")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

PORT = 8501
URL = f"http://localhost:{PORT}"

SAMPLES = {
    "normal": os.path.join(REPO_DIR, "app", "sample_images", "sample_normal.png"),
    "advanced": os.path.join(REPO_DIR, "app", "sample_images", "sample_tb_advanced.png"),
    "moderate": os.path.join(REPO_DIR, "app", "sample_images", "sample_tb_moderate.png"),
    "review": os.path.join(REPO_DIR, "data", "external", "montgomery", "MCUCXR_0162_1.png"),
    "color_ood": "/tmp/test_color_photo.png",
}

# Ensure a synthetic color OOD image exists
if not os.path.exists(SAMPLES["color_ood"]):
    img = Image.new("RGB", (300, 300), color=(255, 60, 40))
    img.save(SAMPLES["color_ood"])

VIEWPORTS = [
    {"name": "desktop", "width": 1440, "height": 1100},
    {"name": "mobile", "width": 390, "height": 950},
]

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

def main():
    print("Checking if Streamlit is already running...")
    started_server = False
    proc = None
    if not wait_for_server(2):
        print(f"Starting Streamlit server on port {PORT}...")
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app/app.py", "--server.port", str(PORT), "--server.headless", "true"],
            cwd=REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        started_server = True
        if not wait_for_server(35):
            print("ERROR: Streamlit failed to start within 35s")
            proc.terminate()
            sys.exit(1)

    print("Streamlit server is ready!")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            for vp in VIEWPORTS:
                device_name = vp["name"]
                print(f"\n==========================================")
                print(f"Capturing screenshots for: {device_name.upper()} ({vp['width']}x{vp['height']})")
                print(f"==========================================")

                context = browser.new_context(viewport={"width": vp["width"], "height": vp["height"]})
                page = context.new_page()
                page.goto(URL, wait_until="networkidle")
                page.wait_for_timeout(2500)

                # On mobile, ensure sidebar overlay is collapsed if open
                if device_name == "mobile":
                    close_btn = page.locator("button[data-testid='stSidebarCollapseButton']")
                    if close_btn.is_visible():
                        try:
                            close_btn.click()
                            page.wait_for_timeout(1000)
                        except Exception:
                            pass

                # 1. Empty State
                print("  1. Empty State...")
                empty_path = os.path.join(SCREENSHOT_DIR, f"{device_name}_empty.png")
                page.screenshot(path=empty_path, full_page=False)

                # 2. Normal Result (Low Risk)
                print("  2. Normal Result (Low Risk)...")
                page.get_by_role("button", name="Normal Chest").click()
                page.wait_for_timeout(3500)
                normal_path = os.path.join(SCREENSHOT_DIR, f"{device_name}_normal.png")
                page.screenshot(path=normal_path, full_page=False)
                if device_name == "desktop":
                    shutil.copy(normal_path, os.path.join(MAIN_DOCS_IMG, "sample_normal_result.png"))

                # 3. High Risk Result
                print("  3. High Risk Result...")
                page.get_by_role("button", name="Advanced TB").click()
                page.wait_for_timeout(3500)
                high_path = os.path.join(SCREENSHOT_DIR, f"{device_name}_high_risk.png")
                page.screenshot(path=high_path, full_page=False)
                if device_name == "desktop":
                    shutil.copy(high_path, os.path.join(MAIN_DOCS_IMG, "sample_tb_heatmap.png"))
                    shutil.copy(high_path, os.path.join(MAIN_DOCS_IMG, "app_overview.png"))

                # 4. Needs Review Result
                print("  4. Needs Review Result...")
                file_input = page.locator("input[type=file]").first
                if os.path.exists(SAMPLES["review"]):
                    file_input.set_input_files(SAMPLES["review"])
                    page.wait_for_timeout(4000)
                    page.screenshot(path=os.path.join(SCREENSHOT_DIR, f"{device_name}_needs_review.png"), full_page=False)

                # 5. Rejected Image (OOD Safety)
                print("  5. Rejected Image...")
                file_input = page.locator("input[type=file]").first
                file_input.set_input_files(SAMPLES["color_ood"])
                page.wait_for_timeout(3000)
                rej_path = os.path.join(SCREENSHOT_DIR, f"{device_name}_rejected.png")
                page.screenshot(path=rej_path, full_page=False)
                if device_name == "desktop":
                    shutil.copy(rej_path, os.path.join(MAIN_DOCS_IMG, "safety_ood_refusal.png"))

                # 6. Batch Tab (4 images)
                print("  6. Batch Tab (4 images)...")
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

                batch_input.set_input_files(batch_files)
                page.wait_for_timeout(2000)

                run_btn = page.get_by_role("button", name="Run Batch Triage Analysis")
                if run_btn.is_visible():
                    run_btn.click()
                    page.wait_for_timeout(5000)

                batch_path = os.path.join(SCREENSHOT_DIR, f"{device_name}_batch.png")
                page.screenshot(path=batch_path, full_page=False)
                if device_name == "desktop":
                    shutil.copy(batch_path, os.path.join(MAIN_DOCS_IMG, "batch_triage_mode.png"))

                context.close()

            browser.close()

        print("\nAll redesign screenshots captured and synced to docs/images/ successfully!")

    finally:
        if started_server and proc is not None:
            print("Stopping Streamlit server...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            print("Server stopped cleanly.")

if __name__ == "__main__":
    main()
