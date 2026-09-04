"""
src/terrain/dem_loader.py
-------------------------
Modèle Numérique de Terrain (DEM / MNT) pour opérations sur le terrain :
- Calcul de la pente locale (slope en degrés et radians)
- Calcul de l'orientation/exposition (aspect)
- Facteur d'accélération topographique de propagation (effet de pente montante)
"""

from typing import Tuple, Optional
import numpy as np
import torch


class DigitalElevationModel:
    """
    Gestionnaire topographique 3D pour la propagation de feu de forêt.
    """

    def __init__(
        self,
        elevation_matrix: np.ndarray,
        resolution_meters: float = 30.0,
        origin_lat_lon: Tuple[float, float] = (43.5, 5.4),  # Exemple Provence / Méditerranée
    ):
        self.elevation = elevation_matrix.astype(np.float32)
        self.dx = resolution_meters
        self.origin = origin_lat_lon
        self.H, self.W = self.elevation.shape

        # Calcul des caractéristiques géomorphologiques
        self.slope_rad, self.aspect_rad = self._compute_slope_and_aspect()
        self.slope_factor = self._compute_slope_acceleration_factor()

    def _compute_slope_and_aspect(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule la pente et l'orientation topographique par différences finies centrales.
        """
        # Gradients d'élévation dz/dx et dz/dy
        dz_dy, dz_dx = np.gradient(self.elevation, self.dx)
        
        # Pente (rad) = arctan(sqrt((dz/dx)^2 + (dz/dy)^2))
        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        
        # Aspect (rad) : direction vers laquelle la pente descend (-pi à +pi)
        aspect_rad = np.arctan2(-dz_dy, dz_dx)
        
        return slope_rad, aspect_rad

    def _compute_slope_acceleration_factor(self) -> np.ndarray:
        """
        Facteur d'accélération de propagation dû à la pente montante (Loi de Rothermel/Albini).
        phi_s = 5.275 * (tan(pente))^2
        """
        tan_slope = np.tan(self.slope_rad)
        phi_s = 5.275 * (tan_slope ** 2)
        return np.clip(phi_s, 0.0, 10.0)  # Borné physiquement

    def get_elevation_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(self.elevation).to(device)

    def get_slope_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.from_numpy(self.slope_rad).to(device)

    def pixel_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convertit les coordonnées pixels locales en coordonnées GPS WGS84.
        """
        lat_origin, lon_origin = self.origin
        # 1 degré de latitude approx 111.32 km
        lat = lat_origin + (y * self.dx) / 111320.0
        # 1 degré de longitude approx 111.32 * cos(lat) km
        lon = lon_origin + (x * self.dx) / (111320.0 * np.cos(np.radians(lat_origin)))
        return float(lat), float(lon)
