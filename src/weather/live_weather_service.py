"""
src/weather/live_weather_service.py
-----------------------------------
Connecteur météorologique et atmosphérique temps réel :
- Extraction des données atmosphériques en direct via API Open-Meteo / Météo-France
- Vitesse à 10m (km/h & m/s), direction du vent, rafales maximales
- Température sous abri 2m, humidité relative (RH), pression atmosphérique
- Calcul de l'humidité du combustible fin mort (FMC) via le modèle d'équilibre de Nelson
- Calcul de l'Indice d'Instabilité Atmosphérique de Haines (risque de feu de cime convectif)
"""

from copy import deepcopy
import threading
import time
from typing import Dict, Any, Tuple
import math
import requests


class LiveWeatherService:
    """
    Gestionnaire météorologique opérationnel pour l'analyse prédictive des feux de forêt.
    """

    _cache_lock = threading.Lock()
    _live_cache: Dict[Tuple[float, float], Tuple[float, Dict[str, Any]]] = {}
    _spatial_cache: Dict[Tuple[float, float, float, int], Tuple[float, Dict[str, Any]]] = {}
    _cache_ttl_sec = 600.0

    def __init__(self):
        self.api_url = "https://api.open-meteo.com/v1/forecast"

    @classmethod
    def _cached(cls, cache: Dict, key, allow_expired: bool = False):
        now = time.monotonic()
        with cls._cache_lock:
            item = cache.get(key)
            if item and (allow_expired or item[0] > now):
                return deepcopy(item[1])
        return None

    @classmethod
    def _store_cache(cls, cache: Dict, key, value: Dict[str, Any]) -> None:
        with cls._cache_lock:
            cache[key] = (time.monotonic() + cls._cache_ttl_sec, deepcopy(value))

    def fetch_live_weather(
        self,
        latitude: float,
        longitude: float,
        timeout_sec: float = 2.5
    ) -> Dict[str, Any]:
        """
        Interroge l'API météorologique pour les coordonnées GPS spécifiées.
        En cas d'indisponibilité réseau, retourne un profil d'été méditerranéen calibré.
        """
        cache_key = (round(float(latitude), 3), round(float(longitude), 3))
        cached = self._cached(self._live_cache, cache_key)
        if cached is not None:
            cached["cache_status"] = "fresh"
            return cached

        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation,rain,showers,cloud_cover",
                "timezone": "auto"
            }
            resp = requests.get(self.api_url, params=params, timeout=timeout_sec)
            if resp.status_code == 200:
                data = resp.json().get("current", {})
                
                temp_c = float(data.get("temperature_2m", 32.5))
                rh = float(data.get("relative_humidity_2m", 28.0))
                wind_speed_kmh = float(data.get("wind_speed_10m", 35.0))
                wind_dir_deg = float(data.get("wind_direction_10m", 315.0))  # 315 = NW (Mistral)
                wind_gust_kmh = float(data.get("wind_gusts_10m", wind_speed_kmh * 1.45))
                pressure_hpa = float(data.get("surface_pressure", 1013.2))
                precip_mm_h = float(data.get("precipitation", data.get("rain", 0.0)))
                cloud_cover = float(data.get("cloud_cover", 10.0))

                wind_speed_ms = wind_speed_kmh / 3.6
                wind_gust_ms = wind_gust_kmh / 3.6
                cardinal_dir = self._deg_to_cardinal(wind_dir_deg)
                
                # Calcul de la composante vectorielle 3D du vent
                rad = math.radians(wind_dir_deg)
                # Convention météo : vent venant de la direction
                u_wind = -wind_speed_ms * math.sin(rad)
                v_wind = -wind_speed_ms * math.cos(rad)
                w_wind = 0.0

                base_fmc = self._compute_nelson_fmc(temp_c, rh)
                # Modélisation de l'humidification rapide par la pluie
                fmc_effective = min(35.0, base_fmc + 6.5 * math.sqrt(precip_mm_h)) if precip_mm_h > 0 else base_fmc
                haines = self._compute_haines_index(temp_c, rh)

                result = {
                    "source": "Open-Meteo API (Temps Réel)",
                    "latitude": latitude,
                    "longitude": longitude,
                    "temperature_c": round(temp_c, 1),
                    "relative_humidity_pct": round(rh, 1),
                    "wind_speed_kmh": round(wind_speed_kmh, 1),
                    "wind_speed_ms": round(wind_speed_ms, 2),
                    "wind_gusts_kmh": round(wind_gust_kmh, 1),
                    "wind_gusts_ms": round(wind_gust_ms, 2),
                    "wind_direction_deg": round(wind_dir_deg, 1),
                    "wind_cardinal": cardinal_dir,
                    "surface_pressure_hpa": round(pressure_hpa, 1),
                    "precipitation_mm_h": round(precip_mm_h, 2),
                    "cloud_cover_pct": round(cloud_cover, 0),
                    "fmc_pct": round(fmc_effective, 1),
                    "base_fmc_pct": round(base_fmc, 1),
                    "haines_index": haines,
                    "wind_vector_3d": [round(u_wind, 2), round(v_wind, 2), round(w_wind, 2)],
                    "is_live_api": True
                }
                self._store_cache(self._live_cache, cache_key, result)
                return result
        except Exception:
            pass  # Bascule en mode hors-ligne calibré

        stale = self._cached(self._live_cache, cache_key, allow_expired=True)
        if stale is not None:
            stale["cache_status"] = "stale_after_provider_error"
            return stale

        # Profil hors-ligne de référence (Conditions estivales à risque FDF)
        temp_c = 34.0
        rh = 22.0
        wind_speed_kmh = 45.0
        wind_dir_deg = 315.0  # Mistral Nord-Ouest
        wind_gust_kmh = 65.0
        precip_mm_h = 0.0
        cloud_cover = 5.0
        wind_speed_ms = wind_speed_kmh / 3.6
        wind_gust_ms = wind_gust_kmh / 3.6

        rad = math.radians(wind_dir_deg)
        u_wind = -wind_speed_ms * math.sin(rad)
        v_wind = -wind_speed_ms * math.cos(rad)

        base_fmc = self._compute_nelson_fmc(temp_c, rh)
        fmc_effective = base_fmc
        haines = self._compute_haines_index(temp_c, rh)

        return {
            "source": "Profil Météorologique Opérationnel de Référence",
            "latitude": latitude,
            "longitude": longitude,
            "temperature_c": temp_c,
            "relative_humidity_pct": rh,
            "wind_speed_kmh": wind_speed_kmh,
            "wind_speed_ms": round(wind_speed_ms, 2),
            "wind_gusts_kmh": wind_gust_kmh,
            "wind_gusts_ms": round(wind_gust_ms, 2),
            "wind_cardinal": "NW",
            "surface_pressure_hpa": 1013.2,
            "precipitation_mm_h": precip_mm_h,
            "cloud_cover_pct": cloud_cover,
            "fmc_pct": round(fmc_effective, 1),
            "base_fmc_pct": round(base_fmc, 1),
            "haines_index": haines,
            "wind_vector_3d": [round(u_wind, 2), round(v_wind, 2), 0.0],
            "is_live_api": False
        }

    def _compute_nelson_fmc(self, temp_c: float, rh_pct: float) -> float:
        """
        Modèle de teneur en eau du combustible fin mort (Nelson 2000 / USFS) :
        FMC = 0.03 + 0.261 * (RH/100) - 0.00114 * (T - 20)
        """
        rh_norm = rh_pct / 100.0
        fmc_fraction = 0.03 + 0.261 * rh_norm - 0.00114 * (temp_c - 20.0)
        fmc_pct = max(3.0, min(30.0, fmc_fraction * 100.0))
        return round(fmc_pct, 1)

    def _compute_haines_index(self, temp_c: float, rh_pct: float) -> Dict[str, Any]:
        """
        Indice de Haines (Indice de sévérité atmosphérique / risque de pyroconvection) :
        Score de 2 à 6 (Faible, Modéré, Élevé, Extrême).
        """
        # Score de stabilité thermique (simplifié selon température de surface)
        if temp_c < 25:
            stability_score = 1
        elif temp_c < 33:
            stability_score = 2
        else:
            stability_score = 3

        # Score d'humidité / sécheresse de l'air
        if rh_pct > 45:
            moisture_score = 1
        elif rh_pct > 25:
            moisture_score = 2
        else:
            moisture_score = 3

        haines_total = stability_score + moisture_score
        
        if haines_total <= 3:
            risk = "Faible"
        elif haines_total == 4:
            risk = "Modéré"
        elif haines_total == 5:
            risk = "Élevé (Potentiel Éruptif)"
        else:
            risk = "EXTRÊME (Risque Pyro-Cumulonimbus / Panache Explosif)"

        return {
            "score": haines_total,
            "level": risk,
            "stability_component": stability_score,
            "moisture_component": moisture_score
        }

    def _deg_to_cardinal(self, deg: float) -> str:
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int((deg + 22.5) / 45.0) % 8
        return dirs[idx]

    def fetch_spatial_weather_grid(
        self,
        latitude: float,
        longitude: float,
        domain_size_m: float,
        sample_grid: int = 5,
        timeout_sec: float = 5.0,
    ) -> Dict[str, Any]:
        """Fetch current weather at several points covering the fire domain."""
        import numpy as np

        n = max(3, min(7, int(sample_grid)))
        cache_key = (round(float(latitude), 3), round(float(longitude), 3), round(float(domain_size_m), 1), n)
        cached = self._cached(self._spatial_cache, cache_key)
        if cached is not None:
            cached["cache_status"] = "fresh"
            return cached
        half = float(domain_size_m) / 2.0
        d_lat = half / 111320.0
        d_lon = d_lat / max(0.2, math.cos(math.radians(latitude)))
        latitudes = np.linspace(latitude - d_lat, latitude + d_lat, n)
        longitudes = np.linspace(longitude - d_lon, longitude + d_lon, n)
        coords = [(float(lat), float(lon)) for lat in latitudes for lon in longitudes]
        params = {
            "latitude": ",".join(str(lat) for lat, _ in coords),
            "longitude": ",".join(str(lon) for _, lon in coords),
            "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m,precipitation,rain,cloud_cover",
            "timezone": "auto",
        }
        try:
            resp = requests.get(self.api_url, params=params, timeout=timeout_sec)
            resp.raise_for_status()
            payload = resp.json()
            records = payload if isinstance(payload, list) else [payload]
            if len(records) != n * n:
                raise ValueError("Open-Meteo spatial response has an unexpected size")

            def grid(field: str, default: float) -> np.ndarray:
                values = []
                for record in records:
                    current = record.get("current", {})
                    value = current.get(field, default)
                    values.append(float(default if value is None else value))
                return np.asarray(values, dtype=np.float32).reshape(n, n)

            speed_kmh = grid("wind_speed_10m", 0.0)
            direction_deg = grid("wind_direction_10m", 315.0)
            direction_rad = np.radians(direction_deg)
            speed_ms = speed_kmh / 3.6
            result = {
                "source": "Open-Meteo current, spatial samples",
                "is_live_api": True,
                "sample_grid": n,
                "latitude": latitudes.round(6).tolist(),
                "longitude": longitudes.round(6).tolist(),
                "temperature_c_grid": grid("temperature_2m", 32.5).round(2).tolist(),
                "relative_humidity_pct_grid": grid("relative_humidity_2m", 28.0).round(2).tolist(),
                "surface_pressure_hpa_grid": grid("surface_pressure", 1013.2).round(2).tolist(),
                "wind_speed_kmh_grid": speed_kmh.round(2).tolist(),
                "wind_direction_deg_grid": direction_deg.round(1).tolist(),
                "wind_gusts_kmh_grid": grid("wind_gusts_10m", 0.0).round(2).tolist(),
                "precipitation_mm_h_grid": grid("precipitation", 0.0).round(3).tolist(),
                "cloud_cover_pct_grid": grid("cloud_cover", 10.0).round(1).tolist(),
                "wind_u_grid": (-speed_ms * np.sin(direction_rad)).round(3).tolist(),
                "wind_v_grid": (-speed_ms * np.cos(direction_rad)).round(3).tolist(),
            }
            self._store_cache(self._spatial_cache, cache_key, result)
            return result
        except Exception:
            stale = self._cached(self._spatial_cache, cache_key, allow_expired=True)
            if stale is not None:
                stale["cache_status"] = "stale_after_provider_error"
                return stale
            return {"source": "spatial weather unavailable", "is_live_api": False, "sample_grid": n}

    def compute_spatial_microclimate_grid(
        self,
        elevation_grid: Any,
        dx_meters: float,
        weather_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calcule les champs atmosphériques 2D/3D distribués et couplés au relief MNT :
        - Gradients altimétriques de température (Lapse Rate 6.5°C/km)
        - Humidité et FMC spatiaux modulés par le versant solaire et les fonds de vallon
        - Accélération topographique du vent sur les crêtes (Effet Venturi / Speedup de crête)
        - Écoulement vertical orographique w(x,y)
        """
        import numpy as np

        elev = np.array(elevation_grid, dtype=np.float32)
        H, W = elev.shape
        mean_elev = float(np.mean(elev))

        # Derive the flow on a smoothed terrain surface. Raw DEM samples can
        # contain stair steps that would otherwise create artificial downdrafts.
        try:
            from scipy.ndimage import gaussian_filter
            terrain_for_flow = gaussian_filter(elev, sigma=1.15, mode="nearest")
        except Exception:
            terrain_for_flow = elev

        # 1. Gradients de terrain
        dz_dy, dz_dx = np.gradient(terrain_for_flow, dx_meters)
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        aspect_rad = np.arctan2(-dz_dy, dz_dx)

        # 2. Données synoptiques de base
        t_ref = float(weather_data.get("temperature_c", 32.0))
        rh_ref = float(weather_data.get("relative_humidity_pct", 25.0))
        w_spd_kmh = float(weather_data.get("wind_speed_kmh", 40.0))
        w_dir_deg = float(weather_data.get("wind_direction_deg", 315.0))
        precip = float(weather_data.get("precipitation_mm_h", 0.0))

        wind_rad = math.radians(w_dir_deg)
        u_syn = -(w_spd_kmh / 3.6) * math.sin(wind_rad)
        v_syn = -(w_spd_kmh / 3.6) * math.cos(wind_rad)

        spatial = weather_data.get("spatial_weather_grid", {})

        def resize_grid(values: Any, default: float) -> np.ndarray:
            source = np.asarray(values, dtype=np.float32) if values is not None else np.asarray([[default]], dtype=np.float32)
            if source.ndim != 2 or source.size == 0:
                source = np.asarray([[default]], dtype=np.float32)
            src_y = np.linspace(0.0, 1.0, source.shape[0])
            src_x = np.linspace(0.0, 1.0, source.shape[1])
            dst_x = np.linspace(0.0, 1.0, W)
            dst_y = np.linspace(0.0, 1.0, H)
            along_x = np.vstack([np.interp(dst_x, src_x, row) for row in source])
            return np.vstack([np.interp(dst_y, src_y, along_x[:, col]) for col in range(W)]).T.astype(np.float32)

        def balance_surface_flux(u_field: np.ndarray, v_field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Remove artificial interior divergence while preserving synoptic inflow."""
            divergence = np.gradient(u_field, dx_meters, axis=1) + np.gradient(v_field, dx_meters, axis=0)
            rhs = divergence - float(np.mean(divergence))
            potential = np.zeros_like(rhs, dtype=np.float32)
            dx2 = float(dx_meters) * float(dx_meters)

            # Neumann edges retain the through-flow at the operational boundary.
            for _ in range(72):
                padded = np.pad(potential, 1, mode="edge")
                potential = 0.25 * (
                    padded[:-2, 1:-1] + padded[2:, 1:-1]
                    + padded[1:-1, :-2] + padded[1:-1, 2:]
                    - dx2 * rhs
                ).astype(np.float32)
                potential -= float(np.mean(potential))

            grad_y, grad_x = np.gradient(potential, dx_meters)
            # Blend the correction to avoid over-fitting the coarse DEM while
            # preserving the measured domain-mean transport vector.
            correction_blend = 0.72
            balanced_u = u_field - correction_blend * grad_x
            balanced_v = v_field - correction_blend * grad_y
            balanced_u += float(np.mean(u_field) - np.mean(balanced_u))
            balanced_v += float(np.mean(v_field) - np.mean(balanced_v))
            return balanced_u.astype(np.float32), balanced_v.astype(np.float32)

        if spatial.get("is_live_api"):
            temp_surface = resize_grid(spatial.get("temperature_c_grid"), t_ref)
            rh_surface = resize_grid(spatial.get("relative_humidity_pct_grid"), rh_ref)
            precip_grid = np.maximum(0.0, resize_grid(spatial.get("precipitation_mm_h_grid"), precip))
            cloud_grid = np.clip(resize_grid(spatial.get("cloud_cover_pct_grid"), 10.0), 0.0, 100.0)
            u_surface = resize_grid(spatial.get("wind_u_grid"), u_syn)
            v_surface = resize_grid(spatial.get("wind_v_grid"), v_syn)
        else:
            temp_surface = np.full((H, W), t_ref, dtype=np.float32)
            rh_surface = np.full((H, W), rh_ref, dtype=np.float32)
            precip_grid = np.full((H, W), max(0.0, precip), dtype=np.float32)
            cloud_grid = np.full((H, W), 10.0, dtype=np.float32)
            u_surface = np.full((H, W), u_syn, dtype=np.float32)
            v_surface = np.full((H, W), v_syn, dtype=np.float32)

        # 3. Champ de Température distribué T(x, y)
        # Gradient altimétrique standard : -0.0065 °C/m + réchauffement solaire des adrets (pentes Sud)
        solar_insolation = np.maximum(0.0, np.cos(aspect_rad - math.radians(180.0))) * np.sin(slope_rad)
        temp_grid = temp_surface - 0.0065 * (elev - mean_elev) + 3.0 * solar_insolation
        temp_grid = np.clip(temp_grid, 5.0, 50.0)

        # 4. Champ d'Humidité Relative RH(x, y)
        # L'air plus frais en altitude a une humidité relative plus élevée
        rh_grid = rh_surface + 0.015 * (elev - mean_elev) - 8.0 * solar_insolation
        rh_grid = np.clip(rh_grid, 8.0, 95.0)

        # 5. Champ de Teneur en Eau du Combustible FMC(x, y)
        rh_norm = rh_grid / 100.0
        fmc_grid = (0.03 + 0.261 * rh_norm - 0.00114 * (temp_grid - 20.0)) * 100.0
        # Effet asséchant de l'exposition solaire
        fmc_grid *= (1.0 - 0.20 * solar_insolation)
        if np.any(precip_grid > 0.0):
            fmc_grid += 6.5 * np.sqrt(precip_grid)
        fmc_grid = np.clip(fmc_grid, 3.0, 35.0)

        # 6. Champ de Vent Topographique 3D U(x, y), V(x, y), W(x, y)
        # Venturi / Speedup de crête : accélération sur les pentes face au vent
        vec_len = np.sqrt(dz_dx**2 + dz_dy**2) + 1e-6
        nx_slope = dz_dx / vec_len
        ny_slope = dz_dy / vec_len
        local_wind_norm = np.sqrt(u_surface**2 + v_surface**2) + 1e-6
        cos_slope_wind = (u_surface * dz_dx + v_surface * dz_dy) / local_wind_norm

        # Facteur d'accélération topographique (jusqu'à +80% sur crêtes)
        speedup = 1.0 + 1.8 * np.maximum(0.0, cos_slope_wind) * np.sin(slope_rad)
        speedup = np.clip(speedup, 0.6, 2.2)

        u_grid, v_grid = balance_surface_flux(u_surface * speedup, v_surface * speedup)
        # Vitesse verticale induite par le relief (forçage orographique)
        # The vertical component is the tangent-plane projection of the
        # horizontal transport over the relief, not an arbitrary updraft.
        w_grid = u_grid * dz_dx + v_grid * dz_dy
        w_grid = np.clip(w_grid, -10.0, 10.0)

        # The 3D scene uses a deliberate 0.75 vertical terrain scale. Keep
        # the physical W field for CFD and expose a display-space companion
        # so vector glyphs remain tangent to the rendered relief.
        terrain_vertical_scale = 0.75
        w_render_grid = u_grid * (dz_dx * terrain_vertical_scale) + v_grid * (dz_dy * terrain_vertical_scale)
        w_render_grid = np.clip(w_render_grid, -10.0, 10.0)

        wind_speed_grid_ms = np.sqrt(u_grid**2 + v_grid**2 + w_grid**2)
        wind_speed_grid_kmh = wind_speed_grid_ms * 3.6

        return {
            "mean_temperature_c": float(np.mean(temp_grid)),
            "min_temperature_c": float(np.min(temp_grid)),
            "max_temperature_c": float(np.max(temp_grid)),
            "mean_rh_pct": float(np.mean(rh_grid)),
            "mean_fmc_pct": float(np.mean(fmc_grid)),
            "max_wind_speed_kmh": float(np.max(wind_speed_grid_kmh)),
            "mean_wind_speed_kmh": float(np.mean(wind_speed_grid_kmh)),
            "temperature_grid": np.round(temp_grid, 1).tolist(),
            "relative_humidity_grid": np.round(rh_grid, 1).tolist(),
            "fmc_grid": np.round(fmc_grid, 1).tolist(),
            "wind_u_grid": np.round(u_grid, 2).tolist(),
            "wind_v_grid": np.round(v_grid, 2).tolist(),
            "wind_w_grid": np.round(w_grid, 2).tolist(),
            "wind_w_render_grid": np.round(w_render_grid, 2).tolist(),
            "wind_speed_kmh_grid": np.round(wind_speed_grid_kmh, 1).tolist(),
            "precipitation_mm_h_grid": np.round(precip_grid, 3).tolist(),
            "cloud_cover_pct_grid": np.round(cloud_grid, 1).tolist(),
            "terrain_height_m_grid": np.round(elev - float(np.min(elev)), 1).tolist(),
            "terrain_height_render_m_grid": np.round((elev - float(np.min(elev))) * terrain_vertical_scale, 1).tolist(),
            "terrain_vertical_scale": terrain_vertical_scale,
            "terrain_gradient_x_grid": np.round(dz_dx, 5).tolist(),
            "terrain_gradient_z_grid": np.round(dz_dy, 5).tolist(),
            "terrain_slope_rad_grid": np.round(slope_rad, 5).tolist(),
            "wind_field_model": "terrain_tangent_surface_mass_balanced_open_meteo",
            "weather_field_source": spatial.get("source", "synoptic fallback"),
            "weather_field_is_live": bool(spatial.get("is_live_api", False))
        }
