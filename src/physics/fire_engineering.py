"""
src/physics/fire_engineering.py
-------------------------------
Moteur d'ingénierie physique et comportement du feu (Normes Rothermel, Byram, Van Wagner, USFS / FDF) :
- Vitesse de propagation de surface R_head, R_flank, R_back (Rothermel 1972)
- Intensité de ligne de feu I_B en kW/m (Byram 1959)
- Longueur de flamme L_f (m) et hauteur de roussissement H_s (m)
- Transition en feu de cime actif / passif (Van Wagner 1977)
- Indice de Difficulté d'Extinction (SDI) & Dimensionnement tactique des moyens (GIFF, Canadair CL-415, Bulldozers)
"""

from typing import Dict, Any, Tuple, List
import math
import heapq
import numpy as np


class FuelModelData:
    """
    Modèles de combustible standardisés (Scott & Burgan 40 / FDF Méditerranée).
    """
    MODELS = {
        "SH5": {
            "name": "Maquis Dense / Garrigue Haute (SH5)",
            "fuel_load_kg_m2": 1.45,       # w0 (kg/m2)
            "fuel_bed_depth_m": 1.80,     # delta (m)
            "sav_ratio_m_inv": 4920.0,    # sigma (1/m)
            "moisture_extinction_pct": 18.0, # M_ext (%)
            "heat_content_kj_kg": 18600.0, # h (kJ/kg)
            "canopy_base_height_m": 3.0,
            "canopy_bulk_density_kg_m3": 0.22,
            "description": "Végétation arbustive haute méditerranéenne très inflammable"
        },
        "TU5": {
            "name": "Pinède d'Alep avec sous-bois dense (TU5)",
            "fuel_load_kg_m2": 1.95,
            "fuel_bed_depth_m": 1.20,
            "sav_ratio_m_inv": 5200.0,
            "moisture_extinction_pct": 22.0,
            "heat_content_kj_kg": 18600.0,
            "canopy_base_height_m": 4.5,
            "canopy_bulk_density_kg_m3": 0.18,
            "description": "Forêt de pins méditerranéens avec fort potentiel de feu de cime"
        },
        "GS2": {
            "name": "Garrigue Basse & Pelouse Sèche (GS2)",
            "fuel_load_kg_m2": 0.65,
            "fuel_bed_depth_m": 0.50,
            "sav_ratio_m_inv": 6200.0,
            "moisture_extinction_pct": 15.0,
            "heat_content_kj_kg": 18000.0,
            "canopy_base_height_m": 10.0,
            "canopy_bulk_density_kg_m3": 0.05,
            "description": "Herbes sèches et buissons bas à propagation initiale rapide"
        },
        "TL8": {
            "name": "Litière d'Aiguilles de Pin / Sous-bois Clair (TL8)",
            "fuel_load_kg_m2": 1.10,
            "fuel_bed_depth_m": 0.15,
            "sav_ratio_m_inv": 4600.0,
            "moisture_extinction_pct": 30.0,
            "heat_content_kj_kg": 18600.0,
            "canopy_base_height_m": 8.0,
            "canopy_bulk_density_kg_m3": 0.12,
            "description": "Litière d'aiguilles dense au sol sous futaie"
        }
    }


class FireBehaviorEngineeringEngine:
    """
    Solveur d'ingénierie physique du feu et d'aide à la décision opérationnelle.
    """

    def __init__(self, fuel_code: str = "SH5"):
        self.fuel_code = fuel_code if fuel_code in FuelModelData.MODELS else "SH5"
        self.fuel_props = FuelModelData.MODELS[self.fuel_code]

    def set_fuel_model(self, fuel_code: str):
        if fuel_code in FuelModelData.MODELS:
            self.fuel_code = fuel_code
            self.fuel_props = FuelModelData.MODELS[fuel_code]

    def compute_fire_behavior(
        self,
        wind_speed_kmh: float,
        slope_deg: float,
        fmc_pct: float,
        air_temp_c: float = 32.0,
        fuel_code_override: str = None
    ) -> Dict[str, Any]:
        """
        Calcule les grandeurs physiques normées de propagation selon Rothermel (1972), Byram (1959) et Van Wagner (1977).
        """
        props = FuelModelData.MODELS.get(fuel_code_override, self.fuel_props)

        w0 = props["fuel_load_kg_m2"]
        delta = props["fuel_bed_depth_m"]
        sigma = props["sav_ratio_m_inv"]
        m_ext = props["moisture_extinction_pct"] / 100.0
        h_comb = props["heat_content_kj_kg"]
        cbh = props["canopy_base_height_m"]
        cbd = props["canopy_bulk_density_kg_m3"]

        mf = max(0.02, min(0.35, fmc_pct / 100.0))
        u_ms = max(0.0, wind_speed_kmh / 3.6)
        u_fpm = u_ms * 196.85  # Conversion m/s -> pieds/min pour équations empiriques Rothermel
        slope_rad = math.radians(abs(slope_deg))

        # 1. Calculs intermédiaires Rothermel
        rho_p = 512.0  # Masse volumique des particules de bois (kg/m3)
        rho_b = w0 / delta  # Densité apparente du lit de combustible
        beta = rho_b / rho_p  # Taux de tassement
        beta_opt = 3.348 * (sigma ** (-0.8189))  # Taux de tassement optimal

        # Vitesse de réaction maximale
        gamma_max = (sigma ** 1.5) / (495.0 + 0.0594 * (sigma ** 1.5))
        A = 133.0 * (sigma ** (-0.7913))
        gamma_prime = gamma_max * ((beta / beta_opt) ** A) * math.exp(A * (1.0 - beta / beta_opt))

        # Facteurs d'atténuation (humidité & minéraux)
        r_m = min(1.0, mf / m_ext)
        eta_m = 1.0 - 2.59 * r_m + 5.11 * (r_m ** 2) - 3.52 * (r_m ** 3)
        eta_m = max(0.0, min(1.0, eta_m))
        eta_s = 0.417  # Atténuation minérale standard

        # Intensité de réaction I_R (kW/m2)
        net_fuel_load = w0 * 0.945  # Déduction des teneurs en silice
        I_R = gamma_prime * net_fuel_load * h_comb * eta_m * eta_s / 60.0  # kW/m2

        # Facteur de flux propagateur xi
        xi = (192.0 + 0.2595 * sigma) ** (-1) * math.exp((0.792 + 0.681 * (sigma ** 0.5)) * (beta + 0.1))

        # Facteur d'accélération du vent Phi_w
        C = 7.47 * (sigma ** (-0.55))
        B = 0.02526 * (sigma ** 0.54)
        E = 0.715 * math.exp(-0.000359 * sigma)
        Phi_w = C * ((u_fpm * 0.4) ** B) * ((beta / beta_opt) ** (-E))

        # Facteur d'accélération de la pente Phi_s
        Phi_s = 5.275 * (beta ** (-0.3)) * (math.tan(slope_rad) ** 2)
        Phi_s = min(15.0, Phi_s)

        # Chaleur de pré-allumage Q_ig (kJ/kg) & Nombre effectif d'échauffement
        epsilon = math.exp(-138.0 / sigma)
        Q_ig = 581.0 + 2594.0 * mf

        # Vitesse de propagation de tête de surface R_head (m/min)
        denom = rho_b * epsilon * Q_ig
        R_head_mpm = (I_R * xi * (1.0 + Phi_w + Phi_s) * 60.0) / denom if denom > 0 else 0.1
        R_head_mpm = max(0.05, min(120.0, R_head_mpm))
        R_head_kmh = R_head_mpm * 0.06

        # Vitesse d'arrière (Backing) et de flanc (Flanking)
        R_back_mpm = (I_R * xi * 60.0) / denom if denom > 0 else 0.02
        length_to_width = max(1.0, 1.0 + 0.25 * (u_ms ** 1.2))
        R_flank_mpm = (R_head_mpm + R_back_mpm) / (2.0 * length_to_width)

        # 2. Intensité de ligne de feu de Byram (kW/m)
        # I_B = H * w_a * R (kW/m) avec R en m/s
        R_head_ms = R_head_mpm / 60.0
        I_Byram = h_comb * w0 * R_head_ms  # kW/m

        # 3. Longueur de flamme L_f (m) selon Byram
        flame_length_m = 0.0775 * (I_Byram ** 0.46)

        # 4. Hauteur de roussissement des cimes H_s (m)
        cooling_factor = math.sqrt(max(1.0, 10.0 - u_ms))
        scorch_height_m = (0.01483 * (I_Byram ** (2.0 / 3.0))) / cooling_factor

        # 5. Transition en feu de cime (Van Wagner 1977)
        I_crit_crown = ((0.010 * cbh * (460.0 + 25.9 * fmc_pct)) ** 1.5)  # kW/m
        is_crowning = I_Byram >= I_crit_crown
        
        if is_crowning:
            crown_status = "FEU DE CIME ACTIF (Propagation Éruptive)"
            R_active_mpm = R_head_mpm * 3.34
            R_active_kmh = R_active_mpm * 0.06
            I_active_byram = I_Byram * 3.5
            flame_length_m = max(flame_length_m, cbh + 8.0)
        else:
            crown_status = "Feu de surface contrôlé (Sous-bois)"
            R_active_mpm = R_head_mpm
            R_active_kmh = R_head_kmh
            I_active_byram = I_Byram

        # 6. Indice de Difficulté d'Extinction (SDI) & Analyse tactique
        if I_active_byram < 350.0:
            sdi_category = "Niveau 1 : Attaque directe au sol (Lances / CCF)"
            suppression_tactic = "Attaque directe sur le front possible avec lances d'attaque FDF."
        elif I_active_byram < 1700.0:
            sdi_category = "Niveau 2 : Attaque aux engins lourds (GIFF / Tranchées)"
            suppression_tactic = "Attaque directe impossible à pied. Nécessite GIFF et création de lignes d'appui."
        elif I_active_byram < 3500.0:
            sdi_category = "Niveau 3 : Attaque Aérienne Requise (Canadair CL-415 / Dash)"
            suppression_tactic = "Intensité critique. Attaque indirecte et largages massifs de retardant obligatoires."
        else:
            sdi_category = "Niveau 4 : EXTRÊME - Incontrôlable / Évacuation Immédiate"
            suppression_tactic = "Comportement éruptif extrême. Repli sur points de survie et évacuation des zones d'impact."

        # 7. Calculateur de dimensionnement tactique des moyens
        # Ligne de feu estimée après 1h (périmètre elliptique P approx pi * (a + b))
        a_axis_m = R_active_mpm * 60.0  # Axe majeur
        b_axis_m = R_flank_mpm * 60.0   # Axe mineur
        perimeter_1h_m = math.pi * (3.0 * (a_axis_m + b_axis_m) - math.sqrt((3*a_axis_m + b_axis_m) * (a_axis_m + 3*b_axis_m)))
        area_1h_ha = (math.pi * a_axis_m * b_axis_m) / 10000.0

        # Ligne active sur le front d'attaque à traiter
        active_front_width_m = 2.0 * b_axis_m
        canadair_drops_needed = max(1, math.ceil(active_front_width_m / 120.0))  # 120m couvert par largage
        retardant_liters_needed = canadair_drops_needed * 6130.0
        giff_needed = max(1, math.ceil(perimeter_1h_m / 800.0))  # 1 GIFF traite approx 800m de ligne stabilisée

        return {
            "fuel_model": props["name"],
            "fuel_code": self.fuel_code,
            "ros_head_mpm": round(R_active_mpm, 2),
            "ros_head_kmh": round(R_active_kmh, 2),
            "ros_flank_mpm": round(R_flank_mpm, 2),
            "ros_back_mpm": round(R_back_mpm, 2),
            "fireline_intensity_kw_m": round(I_active_byram, 1),
            "flame_length_m": round(flame_length_m, 2),
            "scorch_height_m": round(scorch_height_m, 2),
            "critical_crown_intensity_kw_m": round(I_crit_crown, 1),
            "crown_fire_status": crown_status,
            "is_crowning": is_crowning,
            "sdi_category": sdi_category,
            "suppression_tactic": suppression_tactic,
            "projected_1h_area_ha": round(area_1h_ha, 2),
            "projected_1h_perimeter_m": round(perimeter_1h_m, 1),
            "tactical_requirements": {
                "active_front_width_m": round(active_front_width_m, 1),
                "canadair_rotations_needed": canadair_drops_needed,
                "retardant_volume_liters": round(retardant_liters_needed, 0),
                "giff_groups_needed": giff_needed,
                "bulldozer_line_rate_mh": round(active_front_width_m * 1.2, 0)
            }
        }


class TerrainFireSpreadSimulator:
    """
    Simulateur de propagation du front de feu sur terrain 3D réel.
    Calcule la vitesse locale de Rothermel directionnelle selon le vent et la pente,
    puis dérive le front d'onde spatialisé (Fast Marching / Dijkstra 8-voisins).
    """

    def __init__(
        self,
        elevation_grid: np.ndarray,
        dx_meters: float = 30.0,
        fuel_code: str = "SH5"
    ):
        self.H, self.W = elevation_grid.shape
        self.elevation = elevation_grid.astype(np.float32)
        self.dx = dx_meters
        self.fuel_code = fuel_code
        self.engine = FireBehaviorEngineeringEngine(fuel_code)

        # Calcul des gradients de terrain (dz/dy, dz/dx)
        self.dz_dy, self.dz_dx = np.gradient(self.elevation, self.dx)
        self.slope_rad = np.arctan(np.sqrt(self.dz_dx**2 + self.dz_dy**2))
        self.slope_deg = np.degrees(self.slope_rad)

    def simulate(
        self,
        ignition_point_px: Tuple[int, int],
        wind_speed_kmh: float,
        wind_dir_cardinal: str,
        fmc_pct: float,
        fuel_reduction_factor: np.ndarray = None,
        incombustible_mask: np.ndarray = None,
        max_time_minutes: float = 60.0,
        num_output_steps: int = 7
    ) -> Dict[str, Any]:
        """
        Calcule la propagation complète du feu sur la grille MNT et produit les étapes d'évolution.
        """
        dir_angles = {
            "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
            "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0
        }
        wind_deg = dir_angles.get(wind_dir_cardinal, 315.0)
        wind_rad = math.radians(wind_deg)
        u_wind = math.sin(wind_rad)
        v_wind = -math.cos(wind_rad)

        base_phys = self.engine.compute_fire_behavior(
            wind_speed_kmh=wind_speed_kmh,
            slope_deg=0.0,
            fmc_pct=fmc_pct
        )
        r_head_mpm = base_phys["ros_head_mpm"]
        r_back_mpm = base_phys["ros_back_mpm"]
        r_flank_mpm = base_phys["ros_flank_mpm"]
        flame_len_base = base_phys["flame_length_m"]

        t_arr = np.full((self.H, self.W), np.inf, dtype=np.float32)
        
        ig_y, ig_x = ignition_point_px
        ig_y = max(0, min(self.H - 1, int(ig_y)))
        ig_x = max(0, min(self.W - 1, int(ig_x)))
        t_arr[ig_y, ig_x] = 0.0

        pq = [(0.0, ig_y, ig_x)]

        # Stencil isotrope étendu à 16 directions pour éliminer tout biais d'anisotropie de grille (carré/losange)
        # 4 orthogonaux (d = 1.0), 4 diagonaux (d = sqrt(2)), 8 sauts de cavalier (d = sqrt(5))
        sqrt2 = math.sqrt(2.0)
        sqrt5 = math.sqrt(5.0)
        neighbors_16 = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, sqrt2), (-1, 1, sqrt2), (1, -1, sqrt2), (1, 1, sqrt2),
            (-2, -1, sqrt5), (-2, 1, sqrt5), (2, -1, sqrt5), (2, 1, sqrt5),
            (-1, -2, sqrt5), (-1, 2, sqrt5), (1, -2, sqrt5), (1, 2, sqrt5),
        ]

        fuel_reduction = fuel_reduction_factor if fuel_reduction_factor is not None else np.ones((self.H, self.W), dtype=np.float32)
        incombustible = incombustible_mask if incombustible_mask is not None else np.zeros((self.H, self.W), dtype=np.float32)

        # Paramètres d'ellipse de Huygens-Richards (1990)
        a_ell = (r_head_mpm + r_back_mpm) / 2.0
        b_ell = max(0.01, r_flank_mpm)
        c_ell = (r_head_mpm - r_back_mpm) / 2.0
        a2, b2 = a_ell ** 2, b_ell ** 2

        while pq:
            curr_t, y, x = heapq.heappop(pq)
            if curr_t > t_arr[y, x] or curr_t > max_time_minutes:
                continue

            for dy, dx, dist_factor in neighbors_16:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.H and 0 <= nx < self.W:
                    if incombustible[ny, nx] > 0.5:
                        continue

                    dz = self.elevation[ny, nx] - self.elevation[y, x]
                    dist_m = dist_factor * self.dx
                    slope_tan = dz / dist_m
                    slope_factor = 1.0 + 5.275 * max(0.0, slope_tan)**2 if slope_tan > 0 else max(0.2, 1.0 + 1.5 * slope_tan)

                    vec_len = math.sqrt(dx**2 + dy**2)
                    dir_x = dx / vec_len
                    dir_y = dy / vec_len
                    cos_theta = dir_x * u_wind + dir_y * v_wind
                    sin2_theta = max(0.0, 1.0 - cos_theta**2)

                    # Formule polaire exacte d'ellipse de propagation (Richards 1990 / Anderson 1983)
                    denom_ell = a2 * sin2_theta + b2 * (cos_theta**2)
                    if denom_ell > 1e-6:
                        ros_dir = (b2 * (a_ell + c_ell * cos_theta)) / denom_ell
                    else:
                        ros_dir = r_head_mpm

                    ros_eff = max(0.01, ros_dir * slope_factor * fuel_reduction[ny, nx])
                    travel_time_min = dist_m / ros_eff
                    new_t = curr_t + travel_time_min

                    if new_t < t_arr[ny, nx]:
                        t_arr[ny, nx] = new_t
                        heapq.heappush(pq, (new_t, ny, nx))

        time_steps = np.linspace(0.0, max_time_minutes, num_output_steps)
        flame_residence_min = 6.0

        frames = []
        for t_step in time_steps:
            t_val = float(t_step)
            burned_mask = (t_arr <= t_val)
            flaming_mask = burned_mask & (t_arr >= max(0.0, t_val - flame_residence_min))
            char_mask = (t_arr < max(0.0, t_val - flame_residence_min))

            ground_temp = np.full((self.H, self.W), 298.15, dtype=np.float32)
            ground_temp[char_mask] = 450.0
            ground_temp[flaming_mask] = 1450.0

            active_front_pts = []
            if t_val > 0:
                flame_indices = np.argwhere(flaming_mask)
                for py, px in flame_indices:
                    lf_local = flame_len_base * (1.0 + 0.3 * np.sin(self.slope_rad[py, px]))
                    active_front_pts.append([int(px), int(py), round(float(lf_local), 2)])

            burn_count = int(np.sum(burned_mask))
            burn_ha = (burn_count * (self.dx * self.dx)) / 10000.0

            perimeter_px = 0
            if burn_count > 0:
                for py, px in np.argwhere(burned_mask):
                    for dy, dx, _ in neighbors_16[:4]:
                        ny, nx = py + dy, px + dx
                        if not (0 <= ny < self.H and 0 <= nx < self.W) or not burned_mask[ny, nx]:
                            perimeter_px += 1

            perim_km = (perimeter_px * self.dx) / 1000.0

            frames.append({
                "time_min": int(round(t_val)),
                "burned_area_ha": round(burn_ha, 2),
                "perimeter_km": round(perim_km, 2),
                "active_front_length_m": round(len(active_front_pts) * self.dx, 1),
                "active_flame_points": active_front_pts,
                "burned_grid_flat": burned_mask.astype(int).flatten().tolist(),
                "flaming_grid_flat": flaming_mask.astype(int).flatten().tolist(),
                "char_grid_flat": char_mask.astype(int).flatten().tolist()
            })

        return {
            "max_time_minutes": max_time_minutes,
            "arrival_time_grid": np.where(np.isinf(t_arr), -1.0, np.round(t_arr, 1)).tolist(),
            "elevation_grid": self.elevation.tolist(),
            "frames": frames,
            "base_physics": base_phys,
            "final_burned_area_ha": frames[-1]["burned_area_ha"],
            "final_perimeter_km": frames[-1]["perimeter_km"]
        }


class StochasticMonteCarloSpreadSimulator:
    """
    Simulateur Stochastique d'Ensemble Monte-Carlo & Sautes de Feu (Albini / Sardoy) sur MNT réel.
    Calcule des faisceaux de N scénarios stochastiques sous turbulences aérologiques (rafales/direction),
    hétérogénéités d'humidité (FMC) et sautes de feu aléatoires par braises convectives.
    Dérive les enveloppes de risque probabilistes P(brûlé, t) et les intervalles de confiance (CI 90%).
    """

    def __init__(
        self,
        elevation_grid: np.ndarray,
        dx_meters: float = 30.0,
        fuel_code: str = "SH5"
    ):
        self.H, self.W = elevation_grid.shape
        self.elevation = elevation_grid.astype(np.float32)
        self.dx = dx_meters
        self.fuel_code = fuel_code
        self.engine = FireBehaviorEngineeringEngine(fuel_code)

        self.dz_dy, self.dz_dx = np.gradient(self.elevation, self.dx)
        self.slope_rad = np.arctan(np.sqrt(self.dz_dx**2 + self.dz_dy**2))
        self.slope_deg = np.degrees(self.slope_rad)
        self.aspect_rad = np.arctan2(-self.dz_dy, self.dz_dx)

    def simulate_ensemble(
        self,
        ignition_point_px: Tuple[int, int],
        wind_speed_kmh: float,
        wind_dir_cardinal: str,
        fmc_pct: float,
        num_ensembles: int = 30,
        wind_dir_std_deg: float = 18.0,
        wind_speed_std_pct: float = 0.20,
        spotting_enabled: bool = True,
        spotting_max_dist_m: float = 900.0,
        fuel_reduction_factor: np.ndarray = None,
        incombustible_mask: np.ndarray = None,
        precipitation_mm_h: float = 0.0,
        spatial_microclimate: Dict[str, Any] = None,
        max_time_minutes: float = 60.0,
        num_output_steps: int = 7,
        frame_callback=None
    ) -> Dict[str, Any]:
        """
        Exécute le faisceau stochastique Monte-Carlo et génère la distribution probabiliste d'incendie.
        Intègre les champs météo distribués (microclimat 2D/3D, speedup de crête, FMC hétérogène).
        """
        dir_angles = {
            "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
            "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0
        }
        base_wind_deg = dir_angles.get(wind_dir_cardinal, 315.0)

        fuel_reduction = fuel_reduction_factor if fuel_reduction_factor is not None else np.ones((self.H, self.W), dtype=np.float32)
        incombustible = incombustible_mask if incombustible_mask is not None else np.zeros((self.H, self.W), dtype=np.float32)

        # Extraction des champs météorologiques distribués réels si disponibles
        has_microclimate = spatial_microclimate is not None and "wind_u_grid" in spatial_microclimate
        if has_microclimate:
            u_field = np.array(spatial_microclimate["wind_u_grid"], dtype=np.float32)
            v_field = np.array(spatial_microclimate["wind_v_grid"], dtype=np.float32)
            fmc_field = np.array(spatial_microclimate["fmc_grid"], dtype=np.float32)
            w_spd_field = np.array(spatial_microclimate["wind_speed_kmh_grid"], dtype=np.float32)
        else:
            w_rad = math.radians(base_wind_deg)
            u_val = -(wind_speed_kmh / 3.6) * math.sin(w_rad)
            v_val = -(wind_speed_kmh / 3.6) * math.cos(w_rad)
            u_field = np.full((self.H, self.W), u_val, dtype=np.float32)
            v_field = np.full((self.H, self.W), v_val, dtype=np.float32)
            fmc_field = np.full((self.H, self.W), fmc_pct, dtype=np.float32)
            w_spd_field = np.full((self.H, self.W), wind_speed_kmh, dtype=np.float32)

        sqrt2 = math.sqrt(2.0)
        sqrt5 = math.sqrt(5.0)
        neighbors_16 = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, sqrt2), (-1, 1, sqrt2), (1, -1, sqrt2), (1, 1, sqrt2),
            (-2, -1, sqrt5), (-2, 1, sqrt5), (2, -1, sqrt5), (2, 1, sqrt5),
            (-1, -2, sqrt5), (-1, 2, sqrt5), (1, -2, sqrt5), (1, 2, sqrt5),
        ]

        # Grilles d'accumulation d'arrivée pour les N membres Monte Carlo
        ensemble_t_arr = np.full((num_ensembles, self.H, self.W), np.inf, dtype=np.float32)
        all_spot_fires = []  # [[x_land, y_land, time_min, member_idx], ...]

        # Modèle d'assèchement solaire selon l'exposition des versants (Sud plus sec)
        solar_drying = 1.0 - 0.20 * np.cos(self.aspect_rad - math.radians(180.0)) * np.sin(self.slope_rad)
        solar_drying = np.clip(solar_drying, 0.75, 1.25)

        for ens_idx in range(num_ensembles):
            # Le membre 0 est exécuté SANS perturbation stochastique pour former la trajectoire nominale déterministe
            is_nominal = (ens_idx == 0)
            if is_nominal:
                w_dir_k = base_wind_deg
                w_speed_k = max(6.0, wind_speed_kmh)
                fmc_k = fmc_pct
            else:
                # 1. Perturbation stochastique du vent (rafales & turbulences directionnelles)
                w_dir_k = base_wind_deg + float(np.random.normal(0.0, wind_dir_std_deg))
                w_speed_k = max(6.0, wind_speed_kmh * float(1.0 + np.random.normal(0.0, wind_speed_std_pct)))
                # 2. Perturbation stochastique du combustible et de l'humidité
                fmc_k = max(2.5, fmc_pct * float(1.0 + np.random.normal(0.0, 0.12)))

            wind_rad_k = math.radians(w_dir_k)
            u_wind_k = math.sin(wind_rad_k)
            v_wind_k = -math.cos(wind_rad_k)

            phys_k = self.engine.compute_fire_behavior(w_speed_k, 0.0, fmc_k)
            r_head_k = phys_k["ros_head_mpm"]
            r_back_k = phys_k["ros_back_mpm"]
            r_flank_k = phys_k["ros_flank_mpm"]
            byram_k = phys_k["fireline_intensity_kw_m"]

            a_k = (r_head_k + r_back_k) / 2.0
            b_k = max(0.01, r_flank_k)
            c_k = (r_head_k - r_back_k) / 2.0
            a2_k, b2_k = a_k ** 2, b_k ** 2

            t_arr_k = ensemble_t_arr[ens_idx]
            ig_y, ig_x = ignition_point_px
            ig_y = max(0, min(self.H - 1, int(ig_y)))
            ig_x = max(0, min(self.W - 1, int(ig_x)))
            t_arr_k[ig_y, ig_x] = 0.0

            pq = [(0.0, ig_y, ig_x)]

            while pq:
                curr_t, y, x = heapq.heappop(pq)
                if curr_t > t_arr_k[y, x] or curr_t > max_time_minutes:
                    continue

                # Déclenchement stochastique de sautes de feu (Ember spotting - Albini 1979)
                if spotting_enabled and byram_k > 2000.0 and curr_t > 5.0:
                    if np.random.rand() < 0.015:
                        spot_dist_m = float(np.random.weibull(2.2) * (w_speed_k * 15.0))
                        spot_dist_m = min(spotting_max_dist_m, spot_dist_m)
                        
                        if spot_dist_m > 90.0:
                            spot_angle = math.radians(w_dir_k + np.random.normal(0.0, 10.0))
                            dx_spot = int(round((spot_dist_m * math.sin(spot_angle)) / self.dx))
                            dy_spot = int(round((-spot_dist_m * math.cos(spot_angle)) / self.dx))
                            sy, sx = y + dy_spot, x + dx_spot

                            if 0 <= sy < self.H and 0 <= sx < self.W and incombustible[sy, sx] < 0.5:
                                spot_t = curr_t + 1.2
                                if spot_t < t_arr_k[sy, sx]:
                                    t_arr_k[sy, sx] = spot_t
                                    heapq.heappush(pq, (spot_t, sy, sx))
                                    all_spot_fires.append({
                                        "x_px": sx, "y_px": sy,
                                        "time_min": round(spot_t, 1),
                                        "dist_m": round(spot_dist_m, 0),
                                        "member": ens_idx
                                    })

                for dy, dx, dist_factor in neighbors_16:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.H and 0 <= nx < self.W:
                        if incombustible[ny, nx] > 0.5:
                            continue

                        dz = self.elevation[ny, nx] - self.elevation[y, x]
                        dist_m = dist_factor * self.dx
                        slope_tan = dz / dist_m
                        slope_factor = 1.0 + 5.275 * max(0.0, slope_tan)**2 if slope_tan > 0 else max(0.2, 1.0 + 1.5 * slope_tan)

                        vec_len = math.sqrt(dx**2 + dy**2)
                        dir_x = dx / vec_len
                        dir_y = dy / vec_len

                        # Prise en compte du vecteur vent distribué localement (accélération sur crêtes / canalisation)
                        if has_microclimate:
                            u_loc = u_field[ny, nx] * (1.0 + (0.0 if is_nominal else float(np.random.normal(0.0, 0.08))))
                            v_loc = v_field[ny, nx] * (1.0 + (0.0 if is_nominal else float(np.random.normal(0.0, 0.08))))
                            w_spd_loc = max(2.0, math.sqrt(u_loc**2 + v_loc**2))
                            cos_theta = (dir_x * u_loc + dir_y * v_loc) / w_spd_loc
                            fmc_loc = max(2.0, fmc_field[ny, nx] * (1.0 + (0.0 if is_nominal else float(np.random.normal(0.0, 0.05)))))
                        else:
                            cos_theta = dir_x * u_wind_k + dir_y * v_wind_k
                            fmc_loc = fmc_k

                        sin2_theta = max(0.0, 1.0 - cos_theta**2)

                        denom_ell_k = a2_k * sin2_theta + b2_k * (cos_theta**2)
                        if denom_ell_k > 1e-6:
                            ros_dir = (b2_k * (a_k + c_k * cos_theta)) / denom_ell_k
                        else:
                            ros_dir = r_head_k

                        local_fmc_mod = solar_drying[ny, nx] * (fmc_loc / max(2.0, fmc_pct))
                        ros_eff = ros_dir * slope_factor * (1.0 / local_fmc_mod) * fuel_reduction[ny, nx]
                        
                        # Atténuation physique par les précipitations (pluie directe)
                        if precipitation_mm_h > 0.0:
                            rain_factor = math.exp(-0.35 * precipitation_mm_h)
                            ros_eff *= rain_factor
                            if fmc_k > 28.0:
                                ros_eff *= 0.08

                        ros_eff = max(0.01, ros_eff)
                        travel_time_min = dist_m / ros_eff
                        new_t = curr_t + travel_time_min

                        if new_t < t_arr_k[ny, nx]:
                            t_arr_k[ny, nx] = new_t
                            heapq.heappush(pq, (new_t, ny, nx))

        # 3. Synthèse Statistique Spatio-Temporelle pour les Frames de la Timeline
        time_steps = np.linspace(0.0, max_time_minutes, num_output_steps)
        flame_residence_min = 6.0
        frames = []

        base_phys = self.engine.compute_fire_behavior(wind_speed_kmh, 0.0, fmc_pct)
        lf_base = base_phys["flame_length_m"]

        for t_step in time_steps:
            t_val = float(t_step)

            # Matrice 3D binaire [num_ensembles, H, W] : 1 si brûlé à t_val
            ens_burned_3d = (ensemble_t_arr <= t_val)
            
            # Carte de probabilité spatiale P(brûlé, t) = (1/N) * sum(I_burned)
            prob_map = np.mean(ens_burned_3d.astype(np.float32), axis=0)  # Shape [H, W], valeurs entre 0.0 et 1.0

            # Enveloppes de risque probabilistes
            risk_high = (prob_map >= 0.80)   # Zone Rouge : Risque Extrême / Quasi Certain
            risk_med = (prob_map >= 0.50)    # Zone Orange : Forte Probabilité
            risk_low = (prob_map >= 0.20)    # Zone Jaune : Vigilance / Sautes & Rafales
            risk_ext = (prob_map >= 0.05)    # Zone Bleue : Enveloppe Maximale Possible

            # Trajectoire nominale déterministe (Membre 0)
            nominal_burned = ens_burned_3d[0]
            nominal_flaming = nominal_burned & (ensemble_t_arr[0] >= max(0.0, t_val - flame_residence_min))
            nominal_area_ha = float(np.sum(nominal_burned) * (self.dx * self.dx) / 10000.0)

            nominal_flame_pts = []
            if t_val > 0:
                for py, px in np.argwhere(nominal_flaming):
                    lf_nom = lf_base * (1.0 + 0.25 * np.sin(self.slope_rad[py, px]))
                    nominal_flame_pts.append([int(px), int(py), round(float(lf_nom), 2)])

            # Distribution des surfaces brûlées sur l'ensemble
            member_areas_ha = [float(np.sum(ens_burned_3d[i]) * (self.dx * self.dx) / 10000.0) for i in range(num_ensembles)]
            mean_area_ha = float(np.mean(member_areas_ha))
            p10_area_ha = float(np.percentile(member_areas_ha, 10))
            p50_area_ha = float(np.percentile(member_areas_ha, 50))
            p90_area_ha = float(np.percentile(member_areas_ha, 90))
            std_area_ha = float(np.std(member_areas_ha))

            # Périmètre moyen
            mean_perim_km = round(math.sqrt(max(0.1, mean_area_ha)) * 3.54 / 10.0, 2)

            # Extraction des points de front actif probabilistes
            flaming_prob = np.zeros((self.H, self.W), dtype=np.float32)
            if t_val > 0:
                for i in range(num_ensembles):
                    ens_flaming = (ensemble_t_arr[i] <= t_val) & (ensemble_t_arr[i] >= max(0.0, t_val - flame_residence_min))
                    flaming_prob += ens_flaming.astype(np.float32)
                flaming_prob /= float(num_ensembles)

            active_flame_pts = []
            flame_indices = np.argwhere(flaming_prob >= 0.25)

            for py, px in flame_indices:
                prob_val = float(flaming_prob[py, px])
                lf_local = lf_base * (1.0 + 0.25 * np.sin(self.slope_rad[py, px])) * (0.6 + 0.8 * prob_val)
                active_flame_pts.append([int(px), int(py), round(float(lf_local), 2), round(prob_val, 2)])

            # Sautes de feu actives à ce pas de temps
            active_spots = [s for s in all_spot_fires if s["time_min"] <= t_val]

            frames.append({
                "time_min": int(round(t_val)),
                "nominal_area_ha": round(nominal_area_ha, 2),
                "nominal_flame_points": nominal_flame_pts,
                "nominal_burned_flat": nominal_burned.astype(int).flatten().tolist(),
                "mean_area_ha": round(mean_area_ha, 2),
                "p10_area_ha": round(p10_area_ha, 2),
                "p50_area_ha": round(p50_area_ha, 2),
                "p90_area_ha": round(p90_area_ha, 2),
                "std_area_ha": round(std_area_ha, 2),
                "perimeter_km": mean_perim_km,
                "prob_map_flat": (prob_map * 100.0).round().astype(int).flatten().tolist(),
                "risk_high_flat": risk_high.astype(int).flatten().tolist(),
                "risk_med_flat": risk_med.astype(int).flatten().tolist(),
                "risk_low_flat": risk_low.astype(int).flatten().tolist(),
                "active_flame_points": active_flame_pts,
                "spot_fires_count": len(active_spots),
                "spot_fires": active_spots[:30]
            })
            if frame_callback is not None:
                frame_callback(frames[-1])

        # Dimensionnement tactique stochastique (Moyenne et Intervalles de Confiance 90%)
        final_p90 = frames[-1]["p90_area_ha"]
        final_mean = frames[-1]["mean_area_ha"]
        final_p10 = frames[-1]["p10_area_ha"]

        giff_mean = max(1, math.ceil(frames[-1]["perimeter_km"] * 1000.0 / 800.0))
        giff_p90 = max(1, math.ceil(giff_mean * 1.35))
        giff_p10 = max(1, math.ceil(giff_mean * 0.75))

        canadair_mean = max(1, math.ceil(math.sqrt(final_mean * 10000.0) * 1.2 / 120.0))
        canadair_p90 = max(1, math.ceil(math.sqrt(final_p90 * 10000.0) * 1.2 / 120.0))

        return {
            "mode": "STOCHASTIC_MONTE_CARLO",
            "projection_contract": {
                "deterministic": {
                    "name": "nominal_front",
                    "description": "Membre 0 sans perturbation, trajectoire la plus probable conditionnelle aux donnees meteo",
                    "frame_fields": ["nominal_burned_flat", "nominal_flame_points", "nominal_area_ha"],
                },
                "stochastic": {
                    "name": "ensemble_probability",
                    "description": "Distribution des membres perturbes, enveloppes P10/P50/P90 et carte de probabilite",
                    "frame_fields": ["prob_map_flat", "p10_area_ha", "p50_area_ha", "p90_area_ha", "std_area_ha"],
                },
            },
            "num_ensembles": num_ensembles,
            "wind_dir_std_deg": wind_dir_std_deg,
            "wind_speed_std_pct": wind_speed_std_pct,
            "spotting_enabled": spotting_enabled,
            "total_spot_fires_triggered": len(all_spot_fires),
            "max_time_minutes": max_time_minutes,
            "elevation_grid": self.elevation.tolist(),
            "frames": frames,
            "final_mean_area_ha": final_mean,
            "final_ci90_ha": [final_p10, final_p90],
            "tactical_confidence_intervals": {
                "giff_nominal": giff_mean,
                "giff_range": [giff_p10, giff_p90],
                "canadair_nominal": canadair_mean,
                "canadair_range": [canadair_mean, canadair_p90]
            }
        }
