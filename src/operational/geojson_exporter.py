"""
src/operational/geojson_exporter.py
-----------------------------------
Exportation des prédictions en format standard SIG / GIS :
- Contours de front de flamme en GeoJSON (WGS84 EPSG:4326)
- Isochrones de temps d'arrivée et zones de risque
- Directement importable dans QGIS, ArcGIS, ATAK (pompiers/militaires) et Mapbox/Leaflet
"""

import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import numpy as np
import torch
from ..terrain.dem_loader import DigitalElevationModel


class GeoJSONFireExporter:
    """
    Convertit les sorties du tenseur Neural FMM en objets géospatiaux GeoJSON.
    """

    def __init__(self, dem: DigitalElevationModel):
        self.dem = dem

    def export_fire_perimeter_geojson(
        self,
        temperature_grid: torch.Tensor,
        ignition_threshold_norm: float = 0.8,
        time_minutes: int = 60,
        output_path: str = "reports/field_mission/fire_perimeter.geojson"
    ) -> Path:
        """
        Extrait le polygone du périmètre actif et l'enregistre en GeoJSON WGS84.
        """
        temp_2d = temperature_grid[0] if temperature_grid.dim() == 3 else temperature_grid
        mask = (temp_2d.detach().cpu().numpy() > ignition_threshold_norm).astype(np.uint8)

        # Extraction des coordonnées de frontière
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        cnt = ax.contour(mask, levels=[0.5])
        contours = cnt.allsegs[0] if cnt.allsegs else []
        plt.close(fig)

        features = []
        for poly_idx, seg in enumerate(contours):
            if len(seg) < 3:
                continue

            # Conversion pixels -> (Longitude, Latitude)
            coords_wgs84 = []
            for pt in seg:
                x_px, y_px = pt[0], pt[1]
                lat, lon = self.dem.pixel_to_latlon(x_px, y_px)
                coords_wgs84.append([lon, lat])  # Format standard GeoJSON : [Lon, Lat]

            # Fermeture du polygone
            if coords_wgs84[0] != coords_wgs84[-1]:
                coords_wgs84.append(coords_wgs84[0])

            feature = {
                "type": "Feature",
                "properties": {
                    "id": f"front_t_{time_minutes}min_{poly_idx}",
                    "timestamp_minutes": time_minutes,
                    "type": "ACTIVE_FIRE_FRONT",
                    "status": "UNCONTAINED"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords_wgs84]
                }
            }
            features.append(feature)

        geojson_data = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": features
        }

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)

        return out_file

    def export_risk_isochrones_geojson(
        self,
        prob_map_steps: torch.Tensor,
        time_step_minutes: int = 15,
        output_path: str = "reports/field_mission/isochrones.geojson"
    ) -> Path:
        """
        Génère les isochrones de progression du feu pour planification des évacuations.
        """
        num_steps = prob_map_steps.shape[0]
        features = []

        import matplotlib.pyplot as plt

        for s in range(num_steps):
            p_2d = prob_map_steps[s].detach().cpu().numpy()
            contours = plt.contour(p_2d, levels=[0.5]).allsegs[0]
            t_min = (s + 1) * time_step_minutes

            for seg in contours:
                if len(seg) < 3:
                    continue
                coords_wgs84 = []
                for pt in seg:
                    lat, lon = self.dem.pixel_to_latlon(pt[0], pt[1])
                    coords_wgs84.append([lon, lat])

                if coords_wgs84[0] != coords_wgs84[-1]:
                    coords_wgs84.append(coords_wgs84[0])

                feature = {
                    "type": "Feature",
                    "properties": {
                        "isochrone_t_min": t_min,
                        "eta_hours": round(t_min / 60.0, 2),
                        "confidence_risk": "P >= 0.5"
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords_wgs84]
                    }
                }
                features.append(feature)

        plt.close()

        geojson_data = {
            "type": "FeatureCollection",
            "features": features
        }

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)

        return out_file

    def export_probabilistic_risk_geojson(
        self,
        prob_map_2d: np.ndarray,
        spot_fires: List[Dict[str, Any]] = None,
        time_minutes: int = 60,
        output_path: str = "reports/field_mission/probabilistic_risk_wgs84.geojson"
    ) -> Path:
        """
        Génère un GeoJSON multicouches pour les enveloppes de risque probabilistes (P>=0.8, P>=0.5, P>=0.2)
        et les points d'atterrissage des sautes de feu (spotting embers).
        """
        features = []
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        risk_levels = [
            (0.80, "P80_EXTREME", "Zone Rouge : Risque Extrême (P >= 80%)", "#dc2626", 0.60),
            (0.50, "P50_HIGH", "Zone Orange : Forte Probabilité (P >= 50%)", "#ea580c", 0.40),
            (0.20, "P20_MODERATE", "Zone Jaune : Enveloppe de Vigilance (P >= 20%)", "#eab308", 0.25)
        ]

        fig, ax = plt.subplots()
        for threshold, code, label, color, opacity in risk_levels:
            mask = (prob_map_2d >= threshold).astype(np.uint8)
            if np.sum(mask) == 0:
                continue
            cnt = ax.contour(mask, levels=[0.5])
            contours = cnt.allsegs[0] if cnt.allsegs else []

            for poly_idx, seg in enumerate(contours):
                if len(seg) < 3:
                    continue
                coords_wgs84 = []
                for pt in seg:
                    lat, lon = self.dem.pixel_to_latlon(pt[0], pt[1])
                    coords_wgs84.append([lon, lat])

                if coords_wgs84[0] != coords_wgs84[-1]:
                    coords_wgs84.append(coords_wgs84[0])

                features.append({
                    "type": "Feature",
                    "properties": {
                        "risk_code": code,
                        "risk_label": label,
                        "probability_threshold": threshold,
                        "time_minutes": time_minutes,
                        "stroke": color,
                        "fill": color,
                        "fill-opacity": opacity
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [coords_wgs84]
                    }
                })

        plt.close(fig)

        # Ajout des sautes de feu (Spot fires)
        if spot_fires:
            for s_idx, spot in enumerate(spot_fires):
                lx, ly = spot["land_px"]
                lat, lon = self.dem.pixel_to_latlon(lx, ly)
                features.append({
                    "type": "Feature",
                    "properties": {
                        "type": "SPOT_FIRE_EMBER",
                        "time_minutes": spot.get("time_min", 0),
                        "distance_m": spot.get("distance_m", 0),
                        "description": f"Saute de feu secondaire à T+{spot.get('time_min', 0)} min (Distance: {spot.get('distance_m', 0)}m)"
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    }
                })

        geojson_data = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": features
        }

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)

        return out_file
