"""
tests/test_playwright_e2e.py
----------------------------
Test End-to-End (E2E) automatisé avec Playwright pour FireMap Pro (Console C2 / SITAC / CFD) :
1. Démarrage et connexion au serveur FireMap Pro (http://127.0.0.1:5050)
2. Validation du Header C2, badge IA Stochastic FNO + SPH 3D et télémétrie
3. Déclenchement de l'inférence IA Stochastic FNO & Fluide SPH 3D
4. Capture d'écran haute définition de la Vue 3D Three.js MNT / SPH
5. Basculement et capture de la Cartographie SIG 2D Leaflet
6. Basculement et validation de l'onglet Comparaison IA vs Réalité Satellite (Sentinel-2 dNBR / VIIRS)
7. Validation des endpoints d'export (GeoJSON WGS84 et VTK 3D)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import time
import subprocess
import requests
from playwright.sync_api import sync_playwright


def test_playwright_web_gui():
    output_dir = Path("reports/playwright")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("[*] DEBUT DU TEST E2E PLAYWRIGHT - FIREMAP PRO C2 WORKBENCH")
    print("=" * 80)

    server_url = "http://127.0.0.1:5050"
    
    # Nettoyage automatique des processus résiduels sur le port 5050
    try:
        out = subprocess.check_output("netstat -ano | findstr :5050", shell=True).decode()
        for line in out.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) >= 5 and "LISTENING" in parts:
                pid = parts[-1]
                subprocess.run(f"taskkill /F /PID {pid}", shell=True, capture_output=True)
    except Exception:
        pass

    print("[*] Démarrage du serveur FireMap Pro (Threading)...")
    server_process = subprocess.Popen([sys.executable, "-u", "app.py"])
    for i in range(15):
        time.sleep(0.5)
        try:
            r = requests.get(server_url, timeout=1.0)
            if r.status_code == 200:
                print(f"[OK] Serveur prêt en {(i+1)*0.5:.1f}s sur : {server_url}")
                break
        except Exception:
            pass

    with sync_playwright() as p:
        print("[*] Lancement du navigateur Chromium...")
        browser = p.chromium.launch(headless=True, args=["--enable-webgl", "--use-gl=swiftshader", "--no-sandbox"])
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # 1. Navigation
        print(f"[*] Navigation vers {server_url}...")
        page.goto(server_url, wait_until="domcontentloaded")
        page.wait_for_selector("#c2-header", timeout=10000)

        # 2. Validation Titre & Header C2
        assert "FireMap Pro" in page.title(), f"Titre incorrect : {page.title()}"
        header_text = page.inner_text("#c2-header")
        assert "SITAC / FDF5" in header_text, "Badge C2 absent"
        assert "STOCHASTIC FNO" in header_text, "Badge IA FNO absent"
        assert "UTC" in page.inner_text("#top-clock"), "Horloge UTC absente"
        print("[OK] Header de Commandement & Contrôle (C2) et Badge IA validés.")

        # 3. Validation du Canvas 3D Three.js
        canvas_3d = page.locator("#canvas3d canvas")
        canvas_3d.wait_for(state="visible", timeout=10000)
        print("[OK] Canvas WebGL 3D Three.js avec relief MNT initialisé.")

        # 4. Déclenchement de l'inférence IA Stochastic FNO & Fluide SPH 3D
        print("[*] Déclenchement de l'inférence IA Stochastic FNO & SPH 3D...")
        page.click("button:has-text('CALCULER FAISCEAU IA')")

        # Attente de la fin du calcul stochastique & génération du bulletin FDF5
        page.wait_for_function(
            "document.getElementById('briefing-box').innerText.includes('ORDRE') || document.getElementById('briefing-box').innerText.includes('VALIDATION')",
            timeout=60000
        )

        hud_w = page.inner_text("#hud-w")
        hud_temp = page.inner_text("#hud-temp")
        hud_sph = page.inner_text("#hud-sph")
        hud_fno = page.inner_text("#hud-fno-time")
        briefing = page.inner_text("#briefing-box")

        print(f"  -> HUD W Ascendance SPH : {hud_w}")
        print(f"  -> HUD Température      : {hud_temp}")
        print(f"  -> HUD Particules SPH   : {hud_sph}")
        print(f"  -> HUD Inférence FNO    : {hud_fno}")
        assert "ORDRE" in briefing or "VALIDATION" in briefing, "Le bulletin FDF5 n'a pas été généré."

        # 5. Test du lecteur temporel (Timeline) et des modes de visualisation
        time_badge = page.inner_text("#timeline-badge")
        print(f"  -> Badge Timeline : {time_badge}")
        assert "T+" in time_badge, "Badge temporel absent"

        # Test des commutateurs de mode de visualisation (Stochastique / Déterministe / Superposition)
        page.click("#vis-mode-det")
        time.sleep(0.5)
        page.click("#vis-mode-overlay")
        time.sleep(0.5)
        page.click("#vis-mode-stoch")
        time.sleep(0.5)

        # Capture d'écran vue 3D Three.js
        time.sleep(1.0)
        screen_3d = output_dir / "01_firemap_pro_3d_c2_view.png"
        page.screenshot(path=str(screen_3d))
        print(f"[OK] Capture d'écran Vue 3D C2 enregistrée : {screen_3d}")

        # 6. Basculement vers l'onglet Cartographie SIG 2D
        print("[*] Basculement vers l'onglet Cartographie SIG 2D...")
        page.click("#tab-2d-btn")
        time.sleep(1.5)

        map_2d = page.locator("#map2d")
        assert map_2d.is_visible(), "Carte SIG Leaflet non visible."

        screen_2d = output_dir / "02_firemap_pro_2d_sig_view.png"
        page.screenshot(path=str(screen_2d))
        print(f"[OK] Capture d'écran Vue SIG Leaflet enregistrée : {screen_2d}")

        # 7. Basculement vers le 3ème onglet : Comparaison IA vs Réalité Satellite
        print("[*] Basculement vers l'onglet Validation Satellite (Sentinel-2 dNBR)...")
        page.click("#tab-val-btn")
        time.sleep(1.5)

        val_view = page.locator("#validation-view")
        assert val_view.is_visible(), "Vue de validation scientifique non visible."

        val_iou = page.inner_text("#val-iou")
        val_dice = page.inner_text("#val-dice")
        val_prec_rec = page.inner_text("#val-prec-rec")
        val_hausdorff = page.inner_text("#val-hausdorff")
        val_brier = page.inner_text("#val-brier")

        print(f"  -> Scoreboard IoU (Jaccard) : {val_iou}")
        print(f"  -> Scoreboard Dice F1-Score : {val_dice}")
        print(f"  -> Scoreboard Préc / Rappel : {val_prec_rec}")
        print(f"  -> Scoreboard Hausdorff     : {val_hausdorff}")
        print(f"  -> Scoreboard Score Brier   : {val_brier}")

        assert "%" in val_iou, "IoU manquante"
        assert "%" in val_dice, "Dice F1 manquant"

        screen_val = output_dir / "03_firemap_pro_reality_validation_view.png"
        page.screenshot(path=str(screen_val))
        print(f"[OK] Capture d'écran Validation Satellite enregistrée : {screen_val}")

        # 8. Basculement vers le 4ème onglet : Écoulements Thermodynamiques 3D (Matplotlib & Convection)
        print("[*] Basculement vers l'onglet Écoulements 3D (MPL & Convection)...")
        page.click("#tab-thermo-btn")
        time.sleep(1.5)

        thermo_view = page.locator("#thermo-view")
        assert thermo_view.is_visible(), "Vue des écoulements thermodynamiques non visible."
        thermo_img = page.locator("#thermo-mpl-img")
        assert thermo_img.is_visible(), "Image Matplotlib des écoulements 3D non visible."

        screen_thermo = output_dir / "04_firemap_pro_thermodynamic_flows_view.png"
        page.screenshot(path=str(screen_thermo))
        print(f"[OK] Capture d'écran Écoulements 3D enregistrée : {screen_thermo}")

        # 9. Validation des endpoints d'export
        res_vtk = requests.get(f"{server_url}/download/vtk")
        assert res_vtk.status_code == 200, f"Erreur VTK : {res_vtk.status_code}"
        print(f"[OK] Export VTK 3D validé ({len(res_vtk.content)} octets).")

        res_geojson = requests.get(f"{server_url}/download/geojson")
        assert res_geojson.status_code == 200, f"Erreur GeoJSON : {res_geojson.status_code}"
        print(f"[OK] Export GeoJSON WGS84 validé ({len(res_geojson.content)} octets).")

        res_thermo_api = requests.get(f"{server_url}/api/thermodynamic_flows_mpl")
        assert res_thermo_api.status_code == 200, f"Erreur API Thermo MPL : {res_thermo_api.status_code}"
        print(f"[OK] Endpoint API Matplotlib validé ({len(res_thermo_api.content)} octets).")

        browser.close()

    if server_process:
        try:
            server_process.kill()
            server_process.wait(timeout=2.0)
        except Exception:
            pass

    print("=" * 80)
    print("[OK] TEST E2E PLAYWRIGHT FIREMAP PRO VALIDE AVEC SUCCES.")
    print("=" * 80)


if __name__ == "__main__":
    test_playwright_web_gui()
