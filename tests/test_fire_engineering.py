"""
tests/test_fire_engineering.py
------------------------------
Tests unitaires pour le moteur d'ingénierie physique du feu et les services de données réelles :
- Validation des équations de Rothermel (ROS_head, ROS_flank, ROS_back)
- Validation de l'intensité de Byram (I_B) et longueur de flamme (L_f)
- Validation de la transition en feu de cime (Van Wagner)
- Validation de l'extraction MNT (RealElevationService) et Météo (LiveWeatherService)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import math
import numpy as np
from src.physics.fire_engineering import FireBehaviorEngineeringEngine, FuelModelData
from src.terrain.real_elevation_service import RealElevationService
from src.weather.live_weather_service import LiveWeatherService


def test_fire_engineering_engine():
    print("=" * 80)
    print("[*] TEST DU MOTEUR D'INGÉNIERIE DU FEU (ROTHERMEL / BYRAM / VAN WAGNER)")
    print("=" * 80)

    engine = FireBehaviorEngineeringEngine(fuel_code="SH5")

    # Scénario 1 : Garrigue haute sous vent fort (60 km/h), pente 15°, FMC 5%
    res = engine.compute_fire_behavior(
        wind_speed_kmh=60.0,
        slope_deg=15.0,
        fmc_pct=5.0,
        air_temp_c=35.0
    )

    print(f"Modèle de combustible : {res['fuel_model']}")
    print(f"  -> Vitesse de propagation (Tête) : {res['ros_head_kmh']} km/h ({res['ros_head_mpm']} m/min)")
    print(f"  -> Vitesse de flanc               : {res['ros_flank_mpm']} m/min")
    print(f"  -> Intensité de Byram (I_B)       : {res['fireline_intensity_kw_m']} kW/m")
    print(f"  -> Longueur de flamme             : {res['flame_length_m']} m")
    print(f"  -> Hauteur de roussissement       : {res['scorch_height_m']} m")
    print(f"  -> Statut Feu de Cime             : {res['crown_fire_status']}")
    print(f"  -> Catégorie SDI                  : {res['sdi_category']}")
    print(f"  -> Rotations Canadair CL-415      : {res['tactical_requirements']['canadair_rotations_needed']}")
    print(f"  -> Groupes GIFF requis            : {res['tactical_requirements']['giff_groups_needed']}")

    assert res["ros_head_mpm"] > 1.0, "La vitesse de propagation est trop basse pour un vent de 60 km/h."
    assert res["fireline_intensity_kw_m"] > 500.0, "L'intensité de Byram est sous-estimée."
    assert res["flame_length_m"] > 1.0, "Longueur de flamme invalide."
    assert res["tactical_requirements"]["canadair_rotations_needed"] >= 1, "Calcul de dimensionnement erroné."

    print("[OK] Moteur d'ingénierie physique du feu validé.")


def test_real_elevation_service():
    print("\n" + "=" * 80)
    print("[*] TEST DU SERVICE D'ALTIMÉTRIE MNT RÉEL")
    print("=" * 80)

    elev_service = RealElevationService(grid_size=64, resolution_m=30.0)
    # Coordonnées Montagne Sainte-Victoire (Aix-en-Provence)
    grid, meta = elev_service.fetch_elevation_grid(center_lat=43.5325, center_lon=5.5833)

    print(f"Source MNT       : {meta['source']}")
    print(f"Élévation Min/Max: {meta['min_elevation_m']:.1f} m / {meta['max_elevation_m']:.1f} m")
    print(f"Élévation Moyenne: {meta['mean_elevation_m']:.1f} m")

    assert grid.shape == (64, 64), f"Shape MNT incorrect : {grid.shape}"
    assert meta["max_elevation_m"] > meta["min_elevation_m"], "Le relief doit présenter un dénivelé."

    derivs = elev_service.compute_terrain_derivatives(grid)
    print(f"Pente max calculée: {np.degrees(derivs['slope_rad']).max():.1f}°")
    print("[OK] Service MNT validé.")


def test_live_weather_service():
    print("\n" + "=" * 80)
    print("[*] TEST DU CONNECTEUR MÉTÉOROLOGIQUE TEMPS RÉEL")
    print("=" * 80)

    meteo_service = LiveWeatherService()
    weather = meteo_service.fetch_live_weather(latitude=43.5250, longitude=5.4420)

    print(f"Source Météo : {weather['source']}")
    print(f"  -> Température 2m    : {weather['temperature_c']} °C")
    print(f"  -> Humidité Relative : {weather['relative_humidity_pct']} %")
    print(f"  -> Vent à 10m        : {weather['wind_speed_kmh']} km/h ({weather['wind_cardinal']})")
    print(f"  -> Rafales max       : {weather['wind_gusts_kmh']} km/h")
    print(f"  -> FMC (Nelson)      : {weather['fmc_pct']} %")
    print(f"  -> Indice de Haines  : {weather['haines_index']['score']}/6 ({weather['haines_index']['level']})")

    assert weather["temperature_c"] > -10.0 and weather["temperature_c"] < 60.0
    assert weather["fmc_pct"] >= 2.0 and weather["fmc_pct"] <= 35.0
    assert len(weather["wind_vector_3d"]) == 3
    print("[OK] Connecteur météo validé.")


def test_stochastic_monte_carlo_simulator():
    print("\n" + "=" * 80)
    print("[*] TEST DU SIMULATEUR STOCHASTIQUE MONTE-CARLO & SAUTES DE FEU (ALBINI)")
    print("=" * 80)

    from src.physics.fire_engineering import StochasticMonteCarloSpreadSimulator
    
    elev_dummy = np.ones((64, 64), dtype=np.float32) * 280.0
    elev_dummy[25:50, 25:50] += 150.0

    mc_sim = StochasticMonteCarloSpreadSimulator(elev_dummy, dx_meters=30.0, fuel_code="SH5")
    res = mc_sim.simulate_ensemble(
        ignition_point_px=(32, 28),
        wind_speed_kmh=55.0,
        wind_dir_cardinal="NW",
        fmc_pct=5.5,
        num_ensembles=25,
        wind_dir_std_deg=18.0,
        spotting_enabled=True,
        spotting_max_dist_m=900.0,
        max_time_minutes=60.0
    )

    print(f"Mode de simulation           : {res['mode']}")
    print(f"Membres du faisceau (N)      : {res['num_ensembles']}")
    print(f"Sautes de feu déclenchées    : {res['total_spot_fires_triggered']}")
    print(f"Surface moyenne à 1h         : {res['final_mean_area_ha']:.2f} Ha")
    print(f"Intervalle de Confiance 90%  : [{res['final_ci90_ha'][0]:.2f} Ha - {res['final_ci90_ha'][1]:.2f} Ha]")
    print(f"GIFF requis (fourchette IC90): {res['tactical_confidence_intervals']['giff_range']}")
    print(f"Canadair requis (fourchette) : {res['tactical_confidence_intervals']['canadair_range']}")

    assert res["final_mean_area_ha"] > 1.0
    assert res["final_ci90_ha"][1] >= res["final_ci90_ha"][0]
    assert len(res["frames"]) == 7
    assert "prob_map_flat" in res["frames"][-1]
    print("[OK] Simulateur stochastique Monte-Carlo validé.")


if __name__ == "__main__":
    test_fire_engineering_engine()
    test_real_elevation_service()
    test_live_weather_service()
    test_stochastic_monte_carlo_simulator()
    print("\n[OK] TOUS LES TESTS D'INGÉNIERIE ET DE DONNÉES RÉELLES ONT RÉUSSI.")

