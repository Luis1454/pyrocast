"""
src/operational/mission_report.py
---------------------------------
Générateur de Bulletin Tactique Opérationnel & Ordre d'Opérations FDF (CODIS / PC de Crise) :
- Synthèse d'ingénierie physique : Vitesse de propagation (Rothermel), Intensité de Byram, Longueur de flamme
- État du sinistre : Surface active (Ha), Périmètre (km), Risque d'éruption en feu de cime (Van Wagner)
- Analyse de vulnérabilité et Délais d'Impact (ETA) sur les Enjeux et Points Sensibles (POIs)
- Dimensionnement opérationnel des moyens : Groupes GIFF, Rotations Pélicandrome Canadair CL-415, Tranchées Bulldozer
"""

from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
import torch
from ..terrain.dem_loader import DigitalElevationModel
from ..physics.fire_engineering import FireBehaviorEngineeringEngine


class OperationalMissionReporter:
    """
    Rédige la synthèse tactique d'aide à la décision opérationnelle pour le Commandement.
    """

    def __init__(self, dem: DigitalElevationModel, output_dir: str = "reports/field_mission"):
        self.dem = dem
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fire_engine = FireBehaviorEngineeringEngine(fuel_code="SH5")

    def generate_mission_briefing(
        self,
        final_temp_grid: torch.Tensor,
        prob_map_steps: torch.Tensor,
        points_of_interest: List[Dict[str, Any]],
        wind_speed_kmh: float = 45.0,
        wind_direction_cardinal: str = "NW",
        fmc_pct: float = 6.0,
        air_temp_c: float = 34.0,
        ignition_threshold_norm: float = 0.8,
        time_step_minutes: int = 15,
        filename: str = "TACTICAL_MISSION_BRIEFING.txt"
    ) -> Tuple[str, Path]:
        """
        Calcule les grandeurs d'ingénierie et rédige l'Ordre d'Opérations FDF5.
        """
        temp_2d = final_temp_grid[0] if final_temp_grid.dim() == 3 else final_temp_grid
        burn_mask = (temp_2d.detach().cpu().numpy() > ignition_threshold_norm).astype(np.uint8)

        # 1. Surface brûlée (Hectares)
        pixel_area_ha = (self.dem.dx ** 2) / 10000.0
        total_surface_ha = float(np.sum(burn_mask) * pixel_area_ha)

        # 2. Périmètre de feu (km)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cnt = ax.contour(burn_mask, levels=[0.5])
        contours = cnt.allsegs[0] if cnt.allsegs else []
        plt.close(fig)

        total_perimeter_m = 0.0
        for seg in contours:
            for i in range(len(seg) - 1):
                p1, p2 = seg[i], seg[i + 1]
                total_perimeter_m += float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]) * self.dem.dx)
        perimeter_km = total_perimeter_m / 1000.0

        # 3. Calculs d'ingénierie physique du feu (Rothermel & Byram)
        mean_slope_deg = float(np.degrees(self.dem.slope_rad).mean())
        fire_phys = self.fire_engine.compute_fire_behavior(
            wind_speed_kmh=wind_speed_kmh,
            slope_deg=mean_slope_deg,
            fmc_pct=fmc_pct,
            air_temp_c=air_temp_c
        )

        # 4. Calcul de menace et ETA sur les POIs
        poi_alerts = []
        num_steps = prob_map_steps.shape[0]

        for poi in points_of_interest:
            px_x, px_y = poi["coords_px"]
            poi_name = poi["name"]
            lat, lon = self.dem.pixel_to_latlon(px_x, px_y)

            # Recherche du pas de temps d'impact P >= 0.5
            eta_min = None
            for s in range(num_steps):
                p_val = float(prob_map_steps[s, px_y, px_x].item())
                if p_val >= 0.5:
                    eta_min = (s + 1) * time_step_minutes
                    break

            if eta_min is not None:
                poi_alerts.append({
                    "name": poi_name,
                    "coords_gps": (round(lat, 5), round(lon, 5)),
                    "status": "MENACÉ DIRECTEMENT",
                    "eta_minutes": eta_min,
                    "action": f"Évacuation et bouclage requis sous {eta_min} min"
                })
            else:
                poi_alerts.append({
                    "name": poi_name,
                    "coords_gps": (round(lat, 5), round(lon, 5)),
                    "status": "HORS D'ATTEINTE (Horizon simulé)",
                    "eta_minutes": "N/A",
                    "action": "Surveillance préventive"
                })

        # Rédaction de l'Ordre d'Opérations Opérationnel
        briefing_text = f"""================================================================================
           FIREMAP PRO : ORDRE D'OPÉRATIONS & BULLETIN TACTIQUE FDF5
                 Direction des Secours / PC de Crise (CODIS)
================================================================================
[1] SITUATION GÉNÉRALE & CONDITIONS D'AÉROLOGIE
    - Vent dominant         : {wind_direction_cardinal} ({wind_speed_kmh:.1f} km/h) | T_air: {air_temp_c:.1f} °C
    - Humidité Combustible  : {fmc_pct:.1f} % (FMC Nelson) | Pente moyenne: {mean_slope_deg:.1f}°
    - Végétation dominante  : {fire_phys['fuel_model']}

[2] DIAGNOSTIC PHYSIQUE DU COMPORTEMENT DU FEU (ROTHERMEL / BYRAM)
    - Vitesse de tête (ROS) : {fire_phys['ros_head_kmh']:.2f} km/h ({fire_phys['ros_head_mpm']:.1f} m/min)
    - Vitesse de flanc      : {fire_phys['ros_flank_mpm']:.1f} m/min
    - Intensité de Byram    : {fire_phys['fireline_intensity_kw_m']:.1f} kW/m
    - Longueur de flamme    : {fire_phys['flame_length_m']:.2f} m
    - Hauteur d'écobuage    : {fire_phys['scorch_height_m']:.2f} m
    - Risque Feu de Cime    : {fire_phys['crown_fire_status']}
    - Catégorie SDI         : {fire_phys['sdi_category']}

[3] ÉTAT DU SINISTRE (HORIZON SIMULÉ)
    - Surface active brûlée : {total_surface_ha:.2f} Hectares
    - Périmètre actif       : {perimeter_km:.2f} km
    - Potentiel de cime     : {'TRÈS ÉLEVÉ' if fire_phys['is_crowning'] else 'MODÉRÉ'}

[4] ANALYSE DE VULNÉRABILITÉ SUR LES SITES CRITIQUES (POIs)
"""
        for alert in poi_alerts:
            briefing_text += f"""    * {alert['name']} [GPS: {alert['coords_gps'][0]}, {alert['coords_gps'][1]}]
      - Statut             : {alert['status']}
      - Délai d'impact     : {alert['eta_minutes']} min
      - Préconisation      : {alert['action']}
"""

        briefing_text += f"""================================================================================
[5] DIMENSIONNEMENT OPÉRATIONNEL DES MOYENS
    - Moyens terrestres     : {fire_phys['tactical_requirements']['giff_groups_needed']} GIFF recommandés en attaque/protection
    - Moyens aériens        : {fire_phys['tactical_requirements']['canadair_rotations_needed']} rotations Canadair CL-415 ({fire_phys['tactical_requirements']['retardant_volume_liters']} L retardant)
    - Appui génie           : Tranchée coupe-feu {fire_phys['tactical_requirements']['bulldozer_line_rate_mh']} m/h requise
    - Doctrine d'engagement : {fire_phys['suppression_tactic']}
================================================================================
"""

        report_file = self.output_dir / filename
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(briefing_text)

        return briefing_text, report_file
