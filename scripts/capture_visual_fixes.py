#!/usr/bin/env python3
"""Capture visual bugfix verification screenshots with Playwright.

Checks:
1. Icon text check: Expander arrow, upload icon, sidebar arrow, selectbox arrow.
2. Top padding check: Title and Screening Aid badge sit fully below top bar on desktop and mobile.
3. Empty states in EN, HI, TE on desktop and mobile.
4. High-risk result card and layout.
5. Privacy expander open.
"""

import os
import sys
import time
import subprocess
import urllib.request
from playwright.sync_api import sync_playwright

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_DIR, "docs", "images", "visual_fixes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PORT = 8501
URL = f"http://localhost:{PORT}"

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

def inspect_icon_elements(page):
    """Programmatically inspect icon fonts and text rendering."""
    return page.evaluate("""
        () => {
            const icons = document.querySelectorAll(
                '[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded, .material-symbols-outlined, span[class*="material"], [data-testid="stExpanderToggleIcon"] *'
            );
            const results = [];
            icons.forEach(el => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                results.push({
                    tag: el.tagName,
                    text: el.innerText || el.textContent,
                    fontFamily: style.fontFamily,
                    fontSize: style.fontSize,
                    width: rect.width,
                    height: rect.height,
                    visible: rect.width > 0 && rect.height > 0
                });
            });
            return results;
        }
    """)

def main():
    print(f"Starting Streamlit test server on port {PORT}...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app/app.py", "--server.port", str(PORT), "--server.headless", "true"],
        cwd=REPO_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        if not wait_for_server(35):
            print("ERROR: Streamlit server failed to start")
            proc.terminate()
            sys.exit(1)
        print("Streamlit server is ready!")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1050})
            page = context.new_page()
            page.goto(URL, wait_until="networkidle")
            page.wait_for_timeout(2500)

            # =========================================================
            # PART 1: ENGLISH (Desktop & Mobile)
            # =========================================================
            print("\n[Part 1] English Desktop & Mobile Suite...")
            # 1. Desktop Empty State
            print("  - Desktop Empty State (English)...")
            page.screenshot(path=os.path.join(OUTPUT_DIR, "desktop_empty_en.png"), full_page=False)

            # 2. Desktop Privacy Expander Open
            print("  - Desktop Privacy Expander Open...")
            expander_summary = page.locator("summary").first
            if expander_summary.is_visible():
                expander_summary.click()
                page.wait_for_timeout(800)
            page.screenshot(path=os.path.join(OUTPUT_DIR, "desktop_privacy_open.png"), full_page=False)

            # High-resolution crops / zoom-ins
            expander_summary.screenshot(path=os.path.join(OUTPUT_DIR, "zoom_expander_arrow.png"))
            expander_summary.click()  # close
            page.wait_for_timeout(500)

            top_container = page.locator(".block-container").first
            top_container.screenshot(path=os.path.join(OUTPUT_DIR, "zoom_top_header.png"))

            upload_dropzone = page.locator("[data-testid='stFileUploaderDropzone']").first
            if upload_dropzone.is_visible():
                upload_dropzone.screenshot(path=os.path.join(OUTPUT_DIR, "zoom_uploader_icon.png"))

            sidebar_collapse = page.locator("[data-testid='stSidebarCollapseButton']").first
            if sidebar_collapse.is_visible():
                sidebar_collapse.screenshot(path=os.path.join(OUTPUT_DIR, "zoom_sidebar_arrow.png"))

            selectbox_area = page.locator("[data-testid='stSelectbox']").first
            if selectbox_area.is_visible():
                selectbox_area.screenshot(path=os.path.join(OUTPUT_DIR, "zoom_selectbox_arrow.png"))

            # Log computed icon styles
            icons_info = inspect_icon_elements(page)
            print(f"  - Total icon elements inspected: {len(icons_info)}")
            for icon in icons_info[:6]:
                print(f"    * '{icon['text']}': font={icon['fontFamily']}")

            # 3. Desktop High Risk Result
            print("  - Desktop High Risk Result...")
            page.get_by_role("button", name="Advanced TB").click()
            page.wait_for_timeout(3500)
            page.screenshot(path=os.path.join(OUTPUT_DIR, "desktop_high_risk.png"), full_page=False)

            # 4. Mobile (English) - resize to 390x844
            print("  - Mobile High Risk Result (English 390px)...")
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(1000)
            page.screenshot(path=os.path.join(OUTPUT_DIR, "mobile_high_risk.png"), full_page=False)

            # Clear selection on mobile
            btn_clear = page.get_by_role("button", name="Clear Selection")
            if btn_clear.is_visible():
                btn_clear.click()
                page.wait_for_timeout(1000)
            print("  - Mobile Empty State (English 390px)...")
            page.screenshot(path=os.path.join(OUTPUT_DIR, "mobile_empty_en.png"), full_page=False)

            # Mobile Privacy Expander Open
            print("  - Mobile Privacy Expander Open (390px)...")
            mob_exp = page.locator("summary").first
            if mob_exp.is_visible():
                mob_exp.click()
                page.wait_for_timeout(800)
            page.screenshot(path=os.path.join(OUTPUT_DIR, "mobile_privacy_open.png"), full_page=False)
            mob_exp.click()
            page.wait_for_timeout(500)

            # =========================================================
            # PART 2: HINDI (Desktop & Mobile)
            # =========================================================
            print("\n[Part 2] Hindi Desktop & Mobile Suite...")
            page.set_viewport_size({"width": 1440, "height": 1050})
            page.wait_for_timeout(800)

            # Select Hindi in sidebar
            page.locator("[data-testid='stSelectbox']").first.click()
            page.wait_for_timeout(300)
            page.get_by_role("option", name="हिन्दी (Hindi)").click()
            page.wait_for_timeout(2500)

            print("  - Desktop Empty State (Hindi)...")
            page.screenshot(path=os.path.join(OUTPUT_DIR, "desktop_empty_hi.png"), full_page=False)

            # Resize to mobile
            print("  - Mobile Empty State (Hindi 390px)...")
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(1000)
            page.screenshot(path=os.path.join(OUTPUT_DIR, "mobile_empty_hi.png"), full_page=False)

            # =========================================================
            # PART 3: TELUGU (Desktop & Mobile)
            # =========================================================
            print("\n[Part 3] Telugu Desktop & Mobile Suite...")
            page.set_viewport_size({"width": 1440, "height": 1050})
            page.wait_for_timeout(800)

            # Select Telugu in sidebar
            page.locator("[data-testid='stSelectbox']").first.click()
            page.wait_for_timeout(300)
            page.get_by_role("option", name="తెలుగు (Telugu)").click()
            page.wait_for_timeout(2500)

            print("  - Desktop Empty State (Telugu)...")
            page.screenshot(path=os.path.join(OUTPUT_DIR, "desktop_empty_te.png"), full_page=False)

            # Resize to mobile
            print("  - Mobile Empty State (Telugu 390px)...")
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(1000)
            page.screenshot(path=os.path.join(OUTPUT_DIR, "mobile_empty_te.png"), full_page=False)

            context.close()
            browser.close()

            print(f"\nSUCCESS: All visual fix screenshots captured cleanly in {OUTPUT_DIR}!")

    finally:
        print("Stopping test server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("Server stopped cleanly.")

if __name__ == "__main__":
    main()
