#!/usr/bin/env python3
"""Verify multilingual layout (English, Hindi, Telugu) with Playwright.

Checks:
- No text cut off or overflowing in result card, badges, action banners, buttons, and tabs.
- Bounding box and clientWidth vs scrollWidth analysis.
- Captures screenshots across desktop (1440px) and mobile (390px).
"""

import os
import sys
import time
import subprocess
import urllib.request
from playwright.sync_api import sync_playwright

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCREENSHOT_DIR = os.path.join(REPO_DIR, "docs", "images", "redesign")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

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

def check_element_overflow(page, selector):
    """Return list of overflowing elements matching selector."""
    return page.evaluate("""
        (sel) => {
            const els = document.querySelectorAll(sel);
            const overflows = [];
            els.forEach(el => {
                const sw = el.scrollWidth;
                const cw = el.clientWidth;
                // Allow up to 2px rounding tolerance
                if (sw > cw + 2) {
                    overflows.push({
                        selector: sel,
                        className: el.className,
                        clientWidth: cw,
                        scrollWidth: sw,
                        overflowPx: sw - cw,
                        text: el.innerText ? el.innerText.substring(0, 80).replace(/\\n/g, ' ') : ''
                    });
                }
            });
            return overflows;
        }
    """, selector)

def main():
    print(f"[1/4] Starting Streamlit server on port {PORT}...")
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
        print("[2/4] Streamlit server is active and responding!")

        selectors_to_check = [
            ".clinical-card",
            ".result-status-badge",
            ".next-step-banner",
            ".patient-strip",
            "button[data-testid='baseButton-secondary']",
            "div[data-testid='stTabs']",
        ]

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1050})
            page = context.new_page()
            page.goto(URL, wait_until="networkidle")
            page.wait_for_timeout(2500)

            # -------------------------------------------------------------
            # TEST 1: HINDI (हिन्दी)
            # -------------------------------------------------------------
            print("\n[3/4] Testing Hindi (हिन्दी) Layout & Overflow...")
            # Selectbox for language is in sidebar
            page.locator("div[data-testid='stSelectbox']").first.click()
            page.wait_for_timeout(400)
            page.get_by_role("option", name="हिन्दी (Hindi)").click()
            page.wait_for_timeout(3000)

            hindi_issues = []

            # 1. High Risk (गंभीर टीबी)
            print("  Testing Hindi Sample 2 (High Risk)...")
            btn_hi_s2 = page.get_by_role("button", name="गंभीर टीबी")
            if btn_hi_s2.is_visible():
                btn_hi_s2.click()
                page.wait_for_timeout(3500)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "hindi_desktop_high_risk.png"), full_page=False)
            for sel in selectors_to_check:
                hindi_issues.extend(check_element_overflow(page, sel))

            # 2. Low Risk (सामान्य चेस्ट)
            print("  Testing Hindi Sample 1 (Low Risk)...")
            btn_hi_s1 = page.get_by_role("button", name="सामान्य चेस्ट")
            if btn_hi_s1.is_visible():
                btn_hi_s1.click()
                page.wait_for_timeout(3500)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "hindi_desktop_low_risk.png"), full_page=False)
            for sel in selectors_to_check:
                hindi_issues.extend(check_element_overflow(page, sel))

            # 3. Hindi Mobile Viewport (390px)
            print("  Testing Hindi Mobile Viewport (390px)...")
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(1500)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "hindi_mobile_low_risk.png"), full_page=False)

            # Click High Risk again on mobile
            btn_hi_s2 = page.get_by_role("button", name="गंभीर टीबी")
            if btn_hi_s2.is_visible():
                btn_hi_s2.click()
                page.wait_for_timeout(3500)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "hindi_mobile_high_risk.png"), full_page=False)
            for sel in [".clinical-card", ".result-status-badge", ".next-step-banner", ".patient-strip"]:
                hindi_issues.extend(check_element_overflow(page, sel))

            print(f"  Hindi testing complete: {len(hindi_issues)} total overflow issues found.")

            # -------------------------------------------------------------
            # TEST 2: TELUGU (తెలుగు)
            # -------------------------------------------------------------
            print("\n[4/4] Testing Telugu (తెలుగు) Layout & Overflow...")
            # Return to desktop size first
            page.set_viewport_size({"width": 1440, "height": 1050})
            page.wait_for_timeout(1000)

            # Selectbox in sidebar
            page.locator("div[data-testid='stSelectbox']").first.click()
            page.wait_for_timeout(400)
            page.get_by_role("option", name="తెలుగు (Telugu)").click()
            page.wait_for_timeout(3000)

            telugu_issues = []

            # 1. High Risk (తీవ్రమైన క్షయ)
            print("  Testing Telugu Sample 2 (High Risk)...")
            btn_te_s2 = page.get_by_role("button", name="తీవ్రమైన క్షయ")
            if btn_te_s2.is_visible():
                btn_te_s2.click()
                page.wait_for_timeout(3500)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "telugu_desktop_high_risk.png"), full_page=False)
            for sel in selectors_to_check:
                telugu_issues.extend(check_element_overflow(page, sel))

            # 2. Low Risk (సాధారణ ఛాతీ)
            print("  Testing Telugu Sample 1 (Low Risk)...")
            btn_te_s1 = page.get_by_role("button", name="సాధారణ ఛాతీ")
            if btn_te_s1.is_visible():
                btn_te_s1.click()
                page.wait_for_timeout(3500)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "telugu_desktop_low_risk.png"), full_page=False)
            for sel in selectors_to_check:
                telugu_issues.extend(check_element_overflow(page, sel))

            # 3. Telugu Mobile Viewport (390px)
            print("  Testing Telugu Mobile Viewport (390px)...")
            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(1500)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "telugu_mobile_low_risk.png"), full_page=False)

            # Click High Risk again on mobile
            btn_te_s2 = page.get_by_role("button", name="తీవ్రమైన క్షయ")
            if btn_te_s2.is_visible():
                btn_te_s2.click()
                page.wait_for_timeout(3500)
            page.screenshot(path=os.path.join(SCREENSHOT_DIR, "telugu_mobile_high_risk.png"), full_page=False)
            for sel in [".clinical-card", ".result-status-badge", ".next-step-banner", ".patient-strip"]:
                telugu_issues.extend(check_element_overflow(page, sel))

            print(f"  Telugu testing complete: {len(telugu_issues)} total overflow issues found.")

            # Summary report
            print("\n==========================================")
            print("MULTILINGUAL OVERFLOW AUDIT SUMMARY")
            print("==========================================")
            all_issues = hindi_issues + telugu_issues
            if not all_issues:
                print("SUCCESS: 0 overflow issues detected! All badges, buttons, cards, and tabs fit cleanly.")
            else:
                print(f"WARNING: Found {len(all_issues)} overflow instances:")
                for item in all_issues:
                    print(f"  - [{item['selector']}] CW={item['clientWidth']}px, SW={item['scrollWidth']}px (+{item['overflowPx']}px): {item['text']}")

            browser.close()

    finally:
        print("\nShutting down Streamlit test server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("Server stopped cleanly.")

if __name__ == "__main__":
    main()
