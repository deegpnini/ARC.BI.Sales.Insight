"""Captura screenshot do dashboard Streamlit via Playwright.

Uso:
    streamlit run src/visualization/dashboard.py --server.headless true &
    python3 src/visualization/capture_screenshot.py
"""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs" / "dashboard_screenshot.png"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto("http://localhost:8501")
        time.sleep(8)
        page.screenshot(path=str(OUTPUT), full_page=True)
        browser.close()
    print(f"OK: {OUTPUT}")


if __name__ == "__main__":
    main()
