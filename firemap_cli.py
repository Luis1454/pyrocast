"""
firemap_cli.py
--------------
Interface en Ligne de Commande (CLI) pour utilisation opérationnelle sur le terrain.
Usage :
  python firemap_cli.py --lat 43.52 --lon 5.44 --wind-speed 45 --wind-dir NW --fuel garrigue --steps 8
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

from src.terrain.dem_loader import DigitalElevationModel
from src.terrain.fuel_models import FuelModelDatabase
from src.physics.combustion import ThermochemicalCombustionEngine
from src.physics.spotting import StochasticEmberSpottingModel
from src.physics.aerodynamics import CoupledPyroAerodynamics
from src.mesh.octree_graph import AdaptiveOctreeGraphBuilder
from src.models.mgnn_fmm import NeuralFMM_MGNN
from src.inference.monte_carlo import MonteCarloWildfireSimulator
from src.operational.geojson_exporter import GeoJSONFireExporter
from src.operational.tactical_sim import TacticalInterventionManager
from src.operational.mission_report import OperationalMissionReporter


def parse_args():
    parser = argparse.ArgumentParser(description="FireMap CLI - Moteur Tactique de Prédiction Feux de Forêt")
    parser.add_argument("--lat", type=float, default=43.52, help="Latitude GPS du point d'ignition (ex: 43.52)")
    parser.add_argument("--lon", type=float, default=5.44, help="Longitude GPS du point d'ignition (ex: 5.44)")
    parser.add_argument("--wind-speed", type=float, default=40.0, help="Vitesse du vent en km/h")
    parser.add_argument("--wind-dir", type=str, default="NW", choices=["N", "NE", "E", "SE", "S", "SW", "W", "NW"], help="Direction de provenance du vent")
    parser.add_argument("--fuel", type=str, default="garrigue", choices=["grass", "garrigue", "pine", "timber"], help="Type de végétation dominante")
    parser.add_argument("--steps", type=int, default=6, help="Nombre de pas de prévision (15 min par pas)")
    parser.add_argument("--firebreak", action="store_true", help="Activer une tranchée coupe-feu tactique")
    parser.add_argument("--retardant", action="store_true", help="Activer un largage aérien de retardant")
    parser.add_argument("--output-dir", type=str, default="reports/field_mission", help="Dossier d'exportation des résultats")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"[*] FIREMAP TACTICAL CLI - MOTEUR NEURAL FMM / 3D CFD")
    print(f"[*] Coordonnees Ignition : Lat {args.lat:.4f}, Lon {args.lon:.4f}")
    print(f"[*] Meteo : Vent {args.wind_dir} ({args.wind_speed:.1f} km/h) | Vegetation : {args.fuel.upper()}")
    print("=" * 80)

    # 1. Génération topographique DEM 30m
    H, W = 64, 64
    x_grid, y_grid = np.meshgrid(np.linspace(-2, 2, W), np.linspace(-2, 2, H))
    elevation = 250.0 + 120.0 * np.sin(x_grid * 1.2) + 80.0 * np.cos(y_grid * 1.5)  # Relief de colline
    dem = DigitalElevationModel(elevation, resolution_meters=30.0, origin_lat_lon=(args.lat, args.lon))

    # 2. Conversion du vent km/h -> composantes vectorielles 3D (u, v, w) en m/s
    speed_ms = args.wind_speed / 3.6
    dir_angles = {
        "N": (0, -1), "NE": (-0.7, -0.7), "E": (-1, 0), "SE": (-0.7, 0.7),
        "S": (0, 1), "SW": (0.7, 0.7), "W": (1, 0), "NW": (0.7, -0.7)
    }
    dir_u, dir_v = dir_angles[args.wind_dir]
    wind_u = dir_u * speed_ms
    wind_v = dir_v * speed_ms
    wind_w = 0.15 * speed_ms  # Effet de cisaillement vertical initial

    wind_vector_3d = torch.tensor([wind_u, wind_v, wind_w], device=device, dtype=torch.float32)

    # 3. Initialisation de l'état thermo-physique initial [C=7, H, W]
    # C0: T, C1: P, C2: U, C3: V, C4: W, C5: FMC, C6: FireFlux
    state = torch.zeros((7, H, W), device=device)
    
    # Point d'ignition au centre
    r_ign = np.sqrt((x_grid + 0.3)**2 + (y_grid - 0.2)**2)
    flame_init = np.exp(-(r_ign**2) / 0.05) * 2.5
    state[0] = torch.from_numpy(flame_init).to(device)
    state[2] = wind_u
    state[3] = wind_v
    state[4] = wind_w
    state[5] = 0.10  # Humidité du combustible 10%

    # 4. Gestion des barrières tactiques
    tactical_mgr = TacticalInterventionManager((H, W), dx_meters=30.0)
    if args.firebreak:
        print("[*] Application d'une tranchee coupe-feu tactique...")
        tactical_mgr.add_firebreak_line(p1_px=(15, 20), p2_px=(45, 55), width_meters=60.0)
    if args.retardant:
        print("[*] Application d'un largage de retardant par bombardier d'eau...")
        tactical_mgr.add_aerial_retardant_drop(center_px=(35, 30), length_m=300.0)

    state = tactical_mgr.apply_to_state(state)

    # 5. Inférence stochastique Monte-Carlo GPU/CPU
    model = NeuralFMM_MGNN(in_channels=7, hidden_dim=64, out_channels=7, num_layers=3, forcing_dim=3).to(device)
    graph_builder = AdaptiveOctreeGraphBuilder(max_depth=3, min_depth=1, grad_threshold=0.25)
    mc_sim = MonteCarloWildfireSimulator(model, graph_builder, num_ensembles=8 if device.type == "cpu" else 64)

    wind_forecast = wind_vector_3d.unsqueeze(0).expand(args.steps, -1)
    prob_maps = mc_sim.predict_probability_map(state, wind_forecast, steps=args.steps)

    # 6. Exportations SIG GeoJSON & Bulletin de Mission
    exporter = GeoJSONFireExporter(dem)
    reporter = OperationalMissionReporter(dem, output_dir=args.output_dir)

    perimeter_geojson = exporter.export_fire_perimeter_geojson(
        state, output_path=f"{args.output_dir}/perimeter_wgs84.geojson"
    )
    isochrones_geojson = exporter.export_risk_isochrones_geojson(
        prob_maps, time_step_minutes=15, output_path=f"{args.output_dir}/isochrones_evacuation.geojson"
    )

    pois = [
        {"name": "Hameau des Chenes", "coords_px": (45, 25)},
        {"name": "Poste Electrique HT", "coords_px": (20, 50)},
        {"name": "Camping de la Vallee", "coords_px": (50, 48)},
    ]

    briefing, report_file = reporter.generate_mission_briefing(
        state, prob_maps, points_of_interest=pois, wind_speed_kmh=args.wind_speed, wind_direction_cardinal=args.wind_dir
    )

    print(briefing)
    print(f"[OK] Polygones GeoJSON exportes : {perimeter_geojson}")
    print(f"[OK] Isochrones d'evacuation exportees : {isochrones_geojson}")
    print(f"[OK] Bulletin tactique CODIS enregistre : {report_file}")


if __name__ == "__main__":
    main()
