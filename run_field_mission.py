"""
run_field_mission.py
--------------------
Scénario complet de mission tactique sur le terrain :
1. Initialisation topographique MNT (Provence, France)
2. Couplage vent 3D + convection thermique + saut de braises (Spotting)
3. Action tactique de lutte : largage Canadair & tranchée d'arrêt bulldozer
4. Inférence probabiliste Monte-Carlo & calcul d'isochrones
5. Export GeoJSON (QGIS / ATAK) et carte opérationnelle tactique
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.terrain.dem_loader import DigitalElevationModel
from src.physics.combustion import ThermochemicalCombustionEngine
from src.physics.spotting import StochasticEmberSpottingModel
from src.physics.aerodynamics import CoupledPyroAerodynamics
from src.mesh.octree_graph import AdaptiveOctreeGraphBuilder
from src.models.mgnn_fmm import NeuralFMM_MGNN
from src.inference.monte_carlo import MonteCarloWildfireSimulator
from src.operational.geojson_exporter import GeoJSONFireExporter
from src.operational.tactical_sim import TacticalInterventionManager
from src.operational.mission_report import OperationalMissionReporter


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("reports/field_mission")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      LANCEMENT DE LA MISSION TACTIQUE DE TERRAIN - FIREMAP")
    print("=" * 80)

    # 1. Topographie réelle (MNT 64x64 mailles de 30m = 1.92 km x 1.92 km)
    H, W = 64, 64
    x_g, y_g = np.meshgrid(np.linspace(-2, 2, W), np.linspace(-2, 2, H))
    elevation = 320.0 + 150.0 * np.sin(x_g * 1.5) + 90.0 * np.cos(y_g * 1.2) + 40.0 * x_g
    dem = DigitalElevationModel(elevation, resolution_meters=30.0, origin_lat_lon=(43.525, 5.442))

    # 2. Vent météo 3D : Mistral violent (NW, 55 km/h = 15.28 m/s)
    wind_u = 0.707 * 15.28  # Vers le Sud-Est
    wind_v = -0.707 * 15.28
    wind_w = 0.50
    base_wind_3d = torch.tensor([wind_u, wind_v, wind_w], device=device, dtype=torch.float32)

    # 3. État thermo-physique initial [C=7, H, W]
    state = torch.zeros((7, H, W), device=device)
    # Ignition initiale dans le vallon
    r_ign = np.sqrt((x_g + 0.8)**2 + (y_g - 0.7)**2)
    flame_init = np.exp(-(r_ign**2) / 0.03) * 2.5
    state[0] = torch.from_numpy(flame_init).to(device)
    state[2] = wind_u
    state[3] = wind_v
    state[4] = wind_w
    state[5] = 0.08  # Végétation très sèche (FMC 8%)

    # 4. Modules physiques couplés
    combustion_engine = ThermochemicalCombustionEngine()
    spotting_engine = StochasticEmberSpottingModel(dx_meters=30.0, max_spotting_distance_m=600.0)
    aero_engine = CoupledPyroAerodynamics()

    # 5. Intervention tactique des pompiers (Ligne coupe-feu + largage Canadair)
    tactical_mgr = TacticalInterventionManager((H, W), dx_meters=30.0)
    # Tranchée mécanisée sur la crête
    tactical_mgr.add_firebreak_line(p1_px=(25, 10), p2_px=(50, 45), width_meters=60.0)
    # Largage retardant en protection du village
    tactical_mgr.add_aerial_retardant_drop(center_px=(42, 35), length_m=350.0)

    state = tactical_mgr.apply_to_state(state)

    # 6. Inférence Monte-Carlo GPU/CPU
    model = NeuralFMM_MGNN(in_channels=7, hidden_dim=64, out_channels=7, num_layers=3, forcing_dim=3).to(device)
    graph_builder = AdaptiveOctreeGraphBuilder(max_depth=3, min_depth=1, grad_threshold=0.25)
    mc_sim = MonteCarloWildfireSimulator(model, graph_builder, num_ensembles=8 if device.type == "cpu" else 64)

    wind_forecast = base_wind_3d.unsqueeze(0).expand(8, -1)
    prob_maps = mc_sim.predict_probability_map(state, wind_forecast, steps=8)

    # 7. Évaluation des POIs et Bulletin de Mission
    pois = [
        {"name": "Village de Beauregard", "coords_px": (48, 28)},
        {"name": "Ligne Haute Tension 400kV", "coords_px": (30, 45)},
        {"name": "Centre de Secours Avancé", "coords_px": (15, 15)},
    ]

    reporter = OperationalMissionReporter(dem, output_dir=str(output_dir))
    briefing_txt, briefing_file = reporter.generate_mission_briefing(
        state, prob_maps, points_of_interest=pois, wind_speed_kmh=55.0, wind_direction_cardinal="NW"
    )

    # 8. Export GeoJSON pour QGIS / ATAK
    exporter = GeoJSONFireExporter(dem)
    perim_json = exporter.export_fire_perimeter_geojson(state, output_path=str(output_dir / "fire_front_wgs84.geojson"))
    isochr_json = exporter.export_risk_isochrones_geojson(prob_maps, time_step_minutes=15, output_path=str(output_dir / "evacuation_isochrones.geojson"))

    # 9. Carte tactique opérationnelle pour le PC de crise
    fig, ax = plt.subplots(figsize=(11, 9), facecolor="#0e1117")
    ax.set_facecolor("#0a0d14")

    # Fond altimétrique
    topo_im = ax.imshow(dem.elevation, cmap="terrain", origin="lower", alpha=0.6)
    contours_topo = ax.contour(dem.elevation, levels=10, colors="white", alpha=0.25, linewidths=0.8)
    ax.clabel(contours_topo, inline=True, fontsize=8, fmt="%dm")

    # Tracé de la tranchée coupe-feu et retardant
    ax.imshow(np.ma.masked_where(tactical_mgr.incombustible_mask < 0.5, tactical_mgr.incombustible_mask), cmap="Blues", origin="lower", alpha=0.9)
    ax.imshow(np.ma.masked_where(tactical_mgr.fuel_reduction_factor > 0.5, tactical_mgr.fuel_reduction_factor), cmap="Reds", origin="lower", alpha=0.6)

    # Nappe de risque d'incendie (Monte-Carlo)
    p_final = prob_maps[-1].detach().cpu().numpy()
    fire_im = ax.imshow(np.ma.masked_where(p_final < 0.1, p_final), cmap="inferno", origin="lower", alpha=0.85, vmin=0, vmax=1)

    # Isochrones de temps d'arrivée
    iso_lines = ax.contour(p_final, levels=[0.2, 0.5, 0.8], colors=["#00e5ff", "#76ff03", "#ffff00"], linewidths=1.8)
    ax.clabel(iso_lines, inline=True, fontsize=9, fmt="P=%.1f")

    # Placement des POIs
    for p in pois:
        px, py = p["coords_px"]
        ax.plot(px, py, "r^", markersize=10, markeredgecolor="white")
        ax.text(px + 1.2, py, p["name"], color="white", fontsize=10, weight="bold", bbox=dict(facecolor="#1e293b", edgecolor="none", alpha=0.8, pad=2))

    # Flèche de vent
    ax.arrow(10, 54, 10, -10, head_width=2.5, head_length=3, fc="#00e5ff", ec="#00e5ff", width=0.8)
    ax.text(10, 57, "Mistral NW (55 km/h)", color="#00e5ff", fontsize=11, weight="bold")

    ax.set_title("CARTE TACTIQUE OPÉRATIONNELLE DE COMMANDEMENT (CODIS)\nSimulation 3D Couplée : Topographie, Météo & Risque Probabiliste", color="white", fontsize=12, pad=12)
    ax.tick_params(colors="white")

    cbar = plt.colorbar(fire_im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Probabilité de passage du feu P(brûlé)", color="white")

    tactical_map_file = output_dir / "05_tactical_operational_map.png"
    plt.tight_layout()
    plt.savefig(tactical_map_file, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    print(briefing_txt)
    print("=" * 80)
    print(f"[OK] Carte tactique de commandement generee : {tactical_map_file}")
    print(f"[OK] Export GeoJSON front actif              : {perim_json}")
    print(f"[OK] Export GeoJSON isochrones d'evacuation  : {isochr_json}")
    print(f"[OK] Bulletin tactique CODIS enregistre      : {briefing_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
