import subprocess
import sys
import time

import requests
from playwright.sync_api import sync_playwright


server = subprocess.Popen([sys.executable, "-u", "app.py"])
try:
    for _ in range(30):
        try:
            if requests.get("http://127.0.0.1:5050/", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            time.sleep(0.25)
    else:
        raise RuntimeError("serveur FireMap indisponible")

    console_errors = []
    page_errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--enable-webgl", "--use-gl=swiftshader", "--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.goto("http://127.0.0.1:5050/", wait_until="domcontentloaded")
        page.wait_for_selector("#scene-toolbar")
        page.locator("#canvas3d canvas").wait_for(state="visible")
        page.click("#view-reset")
        time.sleep(1)
        page.screenshot(path="reports/playwright/current_ui.png", full_page=True)
        assert page.locator("#scene-toolbar").inner_text()
        assert page.locator("button[data-layer='clouds']").evaluate("el => el.classList.contains('active')")
        page.click("button[data-layer='clouds']")
        assert not page.locator("button[data-layer='clouds']").evaluate("el => el.classList.contains('active')")
        assert not page_errors, page_errors
        print(f"[OK] UI smoke: toolbar 3D active, canvas WebGL visible, console_errors={len(console_errors)}")
        browser.close()
finally:
    server.terminate()
    try:
        server.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
