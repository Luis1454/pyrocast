"""
Couche volumique atmospherique partagee par la prediction et le rendu.

Le solveur transporte des champs scalaires sur une grille eulerienne :
fumee, eau nuageuse, flamme et temperature. Les champs utilisent le meme
champ de vitesse, ce qui evite de dessiner des nuages independants du vent.
Ce module est volontairement separe du rendu : le navigateur ne recoit que
des volumes quantifies, jamais des particules a animer arbitrairement.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass
class AtmosphericVolumeState:
    """Champs scalaires indexes [vertical, horizontal_z, horizontal_x]."""

    temperature_k: np.ndarray
    smoke_density: np.ndarray
    cloud_water: np.ndarray
    flame_density: np.ndarray
    relative_humidity_pct: np.ndarray
    precipitation_mm_h: np.ndarray


class AtmosphericVolumeSolver:
    """Transport eulerien compact pour les couches atmospheriques visibles."""

    def __init__(
        self,
        shape: Tuple[int, int, int] = (16, 48, 48),
        domain_size_m: Tuple[float, float, float] = (6400.0, 2400.0, 6400.0),
        ambient_temp_k: float = 298.15,
    ):
        self.d, self.h, self.w = shape
        self.domain_x_m, self.domain_y_m, self.domain_z_m = domain_size_m
        self.dx = self.domain_x_m / max(1, self.w - 1)
        self.dz = self.domain_z_m / max(1, self.h - 1)
        self.dy = self.domain_y_m / max(1, self.d - 1)
        self.ambient_temp_k = ambient_temp_k
        self.gravity = 9.81
        self.beta = 0.0034
        # Effective sub-grid turbulent diffusivity for the coarse operational
        # grid. It broadens scalar plumes without introducing render-only blobs.
        self.diffusion = 0.10
        self.wind_u = np.zeros(shape, dtype=np.float32)
        self.wind_v = np.zeros(shape, dtype=np.float32)
        self.wind_w = np.zeros(shape, dtype=np.float32)
        self.ground_height_m = np.zeros((self.h, self.w), dtype=np.float32)
        self.state = AtmosphericVolumeState(
            temperature_k=np.full(shape, ambient_temp_k, dtype=np.float32),
            smoke_density=np.zeros(shape, dtype=np.float32),
            cloud_water=np.zeros(shape, dtype=np.float32),
            flame_density=np.zeros(shape, dtype=np.float32),
            relative_humidity_pct=np.full(shape, 50.0, dtype=np.float32),
            precipitation_mm_h=np.zeros(shape, dtype=np.float32),
        )
        self._cloud_equilibrium = np.zeros(shape, dtype=np.float32)
        # Pending fire emissions are held for the solver substeps of the
        # current output interval, then cleared before the next interval.
        self._pending_smoke = np.zeros(shape, dtype=np.float32)
        self._pending_flame = np.zeros(shape, dtype=np.float32)
        self._pending_temperature = np.full(shape, ambient_temp_k, dtype=np.float32)

    def set_wind_field(self, wind_u_surface: np.ndarray, wind_v_surface: np.ndarray, wind_w_surface: np.ndarray = None, gust_factor: float = 1.0) -> None:
        """Build one vertically extended wind field shared by every scalar."""
        u_surface = self._resample_surface(wind_u_surface, 0.0)
        v_surface = self._resample_surface(wind_v_surface, 0.0)
        w_surface = self._resample_surface(wind_w_surface, 0.0) if wind_w_surface is not None else np.zeros((self.h, self.w), dtype=np.float32)
        altitude = np.arange(self.d, dtype=np.float32)[:, None, None] * self.dy
        available_height = np.maximum(self.domain_y_m - self.ground_height_m, self.dy)[None, :, :]
        above_ground = np.clip((altitude - self.ground_height_m[None, :, :]) / available_height, 0.0, 1.0)
        levels = 0.55 + 0.45 * above_ground
        self.wind_u[:] = u_surface[None, :, :] * levels * float(gust_factor)
        self.wind_v[:] = v_surface[None, :, :] * levels * float(gust_factor)
        self.wind_w[:] = w_surface[None, :, :] * levels
        self._apply_ground_mask()

    def set_wind(self, wind_u_ms: float, wind_v_ms: float, gust_factor: float = 1.0) -> None:
        """Build a fallback uniform wind field when no spatial weather is available."""
        self.set_wind_field(
            np.full((self.h, self.w), float(wind_u_ms), dtype=np.float32),
            np.full((self.h, self.w), float(wind_v_ms), dtype=np.float32),
            gust_factor=gust_factor,
        )

    def _resample_surface(self, values: np.ndarray, default: float, height: int = None, width: int = None) -> np.ndarray:
        height = self.h if height is None else int(height)
        width = self.w if width is None else int(width)
        source = np.asarray(values, dtype=np.float32) if values is not None else np.asarray([[default]], dtype=np.float32)
        if source.ndim != 2 or source.size == 0:
            source = np.asarray([[default]], dtype=np.float32)
        y = np.linspace(0.0, source.shape[0] - 1.0, height, dtype=np.float32)
        x = np.linspace(0.0, source.shape[1] - 1.0, width, dtype=np.float32)
        yy, xx = np.meshgrid(y, x, indexing="ij")
        return map_coordinates(source, [yy, xx], order=1, mode="nearest").astype(np.float32)

    def set_meteorological_fields(self, microclimate: Dict[str, object]) -> None:
        """Project real surface weather fields into the 3D transport domain."""
        temp = self._resample_surface(microclimate.get("temperature_grid"), 25.0, self.h, self.w)
        humidity = self._resample_surface(microclimate.get("relative_humidity_grid"), 50.0, self.h, self.w)
        rain = np.maximum(0.0, self._resample_surface(microclimate.get("precipitation_mm_h_grid"), 0.0, self.h, self.w))
        cloud_cover = np.clip(self._resample_surface(microclimate.get("cloud_cover_pct_grid"), 10.0, self.h, self.w), 0.0, 100.0)
        self.ground_height_m = np.clip(
            self._resample_surface(
                microclimate.get("terrain_height_render_m_grid"),
                0.0,
                self.h,
                self.w,
            ),
            0.0,
            max(0.0, self.domain_y_m - self.dy),
        )
        self.set_wind_field(
            microclimate.get("wind_u_grid"),
            microclimate.get("wind_v_grid"),
            microclimate.get("wind_w_grid"),
        )

        altitude = np.arange(self.d, dtype=np.float32)[:, None, None] * self.dy
        self.state.temperature_k[:] = (temp[None, :, :] + 273.15 - 0.0065 * altitude).astype(np.float32)
        self.state.relative_humidity_pct[:] = np.clip(humidity[None, :, :] + 0.004 * altitude, 5.0, 100.0)
        self.state.precipitation_mm_h[:] = rain[None, :, :]

        humidity_activation = np.clip((humidity - 35.0) / 65.0, 0.0, 1.0)
        activation = (cloud_cover / 100.0) * (0.35 + 0.65 * humidity_activation)
        cloud_base = 720.0 + (1.0 - float(np.mean(humidity_activation))) * 520.0
        vertical = np.arange(self.d, dtype=np.float32)[:, None, None] * self.dy
        layer = np.exp(-0.5 * ((vertical - cloud_base) / 150.0) ** 2)
        self._cloud_equilibrium[:] = (layer * activation[None, :, :]).astype(np.float32)
        self.state.cloud_water[:] = self._cloud_equilibrium
        self._apply_ground_mask()

    def initialize_cloud_layer(self, cloud_cover_pct: float, relative_humidity_pct: float) -> None:
        """Initialise un condensat nuageux horizontal, puis le laisse etre transporte."""
        coverage = np.clip(float(cloud_cover_pct) / 100.0, 0.0, 1.0)
        humidity = np.clip((float(relative_humidity_pct) - 35.0) / 65.0, 0.0, 1.0)
        activation = coverage * (0.35 + 0.65 * humidity)
        vertical = np.arange(self.d, dtype=np.float32)[:, None, None] * self.dy
        cloud_base = 720.0 + (1.0 - humidity) * 520.0
        layer = np.exp(-0.5 * ((vertical - cloud_base) / 150.0) ** 2)
        x = np.linspace(-1.0, 1.0, self.w, dtype=np.float32)[None, None, :]
        z = np.linspace(-1.0, 1.0, self.h, dtype=np.float32)[None, :, None]
        horizontal = np.clip(1.0 - 0.20 * (x * x + z * z), 0.55, 1.0)
        self._cloud_equilibrium[:] = (activation * layer * horizontal).astype(np.float32)
        self.state.cloud_water[:] = self._cloud_equilibrium
        self._apply_ground_mask()

    def inject_fire_sources(self, sources: Iterable[Sequence[float]]) -> None:
        """Injecte chaleur, flamme et suie dans le volume autour du front actif."""
        self._pending_smoke.fill(0.0)
        self._pending_flame.fill(0.0)
        self._pending_temperature.fill(self.ambient_temp_k)
        if not sources:
            return
        x_axis = np.linspace(-self.domain_x_m / 2.0, self.domain_x_m / 2.0, self.w, dtype=np.float32)
        z_axis = np.linspace(-self.domain_z_m / 2.0, self.domain_z_m / 2.0, self.h, dtype=np.float32)
        y_axis = np.arange(self.d, dtype=np.float32) * self.dy
        zz, yy, xx = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
        # meshgrid ci-dessus est [H,D,W]; les champs sont [D,H,W].
        xx = np.transpose(xx, (1, 0, 2))
        yy = np.transpose(yy, (1, 0, 2))
        zz = np.transpose(zz, (1, 0, 2))

        for source in sources:
            if len(source) < 3:
                continue
            sx, sy, sz = float(source[0]), float(source[1]), float(source[2])
            flame_height = max(2.0, float(source[3]) if len(source) > 3 else 8.0)
            probability = float(source[4]) if len(source) > 4 else 1.0
            # A coarse cell is an areal average, so the source footprint must
            # cover more than one cell before advection and diffusion act.
            horizontal_sigma = max(self.dx * 1.8, 45.0)
            vertical_sigma = max(self.dy * 1.0, flame_height * 0.55)
            kernel = np.exp(
                -0.5 * (((xx - sx) / horizontal_sigma) ** 2 + ((zz - sz) / horizontal_sigma) ** 2)
            ) * np.exp(-0.5 * (((yy - sy - flame_height * 0.35) / vertical_sigma) ** 2))
            strength = np.clip(probability, 0.1, 1.0)
            source_flame = kernel * strength
            source_smoke = source_flame * 0.72
            source_temperature = self.ambient_temp_k + kernel * (1150.0 + 90.0 * flame_height)
            self._pending_flame[:] = np.maximum(self._pending_flame, source_flame)
            self._pending_smoke[:] = np.maximum(self._pending_smoke, source_smoke)
            self._pending_temperature[:] = np.maximum(self._pending_temperature, source_temperature)
            self.state.flame_density[:] = np.maximum(self.state.flame_density, source_flame)
            self.state.smoke_density[:] = np.maximum(self.state.smoke_density, source_smoke)
            self.state.temperature_k[:] = np.maximum(self.state.temperature_k, source_temperature)
        self._apply_ground_mask()

    def _apply_pending_fire_sources(self) -> None:
        """Maintain the current front emission while a frame is sub-stepped."""
        if not np.any(self._pending_flame):
            return
        self.state.flame_density = np.maximum(self.state.flame_density, self._pending_flame).astype(np.float32)
        self.state.smoke_density = np.maximum(self.state.smoke_density, self._pending_smoke).astype(np.float32)
        self.state.temperature_k = np.maximum(self.state.temperature_k, self._pending_temperature).astype(np.float32)

    @staticmethod
    def _diffuse(field: np.ndarray, amount: float) -> np.ndarray:
        padded = np.pad(field, 1, mode="edge")
        neighbours = (
            padded[:-2, 1:-1, 1:-1] + padded[2:, 1:-1, 1:-1]
            + padded[1:-1, :-2, 1:-1] + padded[1:-1, 2:, 1:-1]
            + padded[1:-1, 1:-1, :-2] + padded[1:-1, 1:-1, 2:]
        ) / 6.0
        result = field + amount * (neighbours - field)
        result[0] = field[0]
        result[-1] = field[-1]
        return result

    def _advect(self, field: np.ndarray, dt_s: float) -> np.ndarray:
        d, h, w = field.shape
        zi, yi, xi = np.meshgrid(
            np.arange(d, dtype=np.float32),
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing="ij",
        )
        source = [
            zi - self.wind_w * dt_s / self.dy,
            yi - self.wind_v * dt_s / self.dz,
            xi - self.wind_u * dt_s / self.dx,
        ]
        # Open boundaries prevent smoke and cloud mass from being mirrored or
        # pinned at the edge of the operational domain.
        return map_coordinates(field, source, order=1, mode="constant", cval=0.0).astype(np.float32)

    def _apply_ground_mask(self) -> None:
        """Keep all transported scalars and velocity below the MNT inactive."""
        vertical = np.arange(self.d, dtype=np.float32)[:, None, None] * self.dy
        solid = vertical < self.ground_height_m[None, :, :]
        self.wind_u = np.where(solid, 0.0, self.wind_u).astype(np.float32)
        self.wind_v = np.where(solid, 0.0, self.wind_v).astype(np.float32)
        self.wind_w = np.where(solid, 0.0, self.wind_w).astype(np.float32)
        self.state.smoke_density = np.where(solid, 0.0, self.state.smoke_density).astype(np.float32)
        self.state.cloud_water = np.where(solid, 0.0, self.state.cloud_water).astype(np.float32)
        self.state.flame_density = np.where(solid, 0.0, self.state.flame_density).astype(np.float32)
        self.state.temperature_k = np.where(solid, self.ambient_temp_k, self.state.temperature_k).astype(np.float32)
        self.state.relative_humidity_pct = np.where(solid, 100.0, self.state.relative_humidity_pct).astype(np.float32)
        self.state.precipitation_mm_h = np.where(solid, 0.0, self.state.precipitation_mm_h).astype(np.float32)
        self._cloud_equilibrium = np.where(solid, 0.0, self._cloud_equilibrium).astype(np.float32)

    def step(self, dt_s: float = 2.0, substeps: int = 1) -> None:
        """Avance les champs par advection, diffusion, flottabilite et decroissance."""
        dt = float(dt_s) / max(1, int(substeps))
        for _ in range(max(1, int(substeps))):
            self._apply_pending_fire_sources()
            temperature_excess = np.maximum(0.0, self.state.temperature_k - self.ambient_temp_k)
            self.wind_w += np.clip(self.gravity * self.beta * temperature_excess * dt, 0.0, 26.0)
            self.wind_w *= 0.985

            temperature = self._advect(self.state.temperature_k, dt)
            smoke = self._advect(self.state.smoke_density, dt)
            cloud = self._advect(self.state.cloud_water, dt)
            flame = self._advect(self.state.flame_density, dt)

            # Move the condensation reservoir with the same velocity field as
            # the cloud water. A fixed equilibrium layer would make clouds
            # visibly reappear against the measured wind direction.
            self._cloud_equilibrium = self._advect(self._cloud_equilibrium, dt)
            self._cloud_equilibrium = np.clip(
                self._diffuse(self._cloud_equilibrium, self.diffusion * 0.35),
                0.0,
                1.0,
            )

            smoke += flame * min(0.22, dt / 120.0)
            temperature += flame * 10.0
            cloud += (self._cloud_equilibrium - cloud) * min(0.08, dt / 900.0)

            # Rain removes aerosol mass; this is a volume sink, not a visual toggle.
            rain = self._advect(self.state.precipitation_mm_h, dt)
            smoke *= np.exp(-np.clip(rain, 0.0, 80.0) * dt / 3600.0 * 0.65)
            cloud *= np.exp(-np.clip(rain, 0.0, 80.0) * dt / 3600.0 * 0.18)

            self.state.temperature_k = np.maximum(self.ambient_temp_k, self._diffuse(temperature, self.diffusion * 0.35))
            self.state.smoke_density = np.clip(self._diffuse(smoke, self.diffusion), 0.0, 1.0)
            self.state.cloud_water = np.clip(self._diffuse(cloud, self.diffusion * 0.7), 0.0, 1.0)
            self.state.flame_density = np.clip(flame * np.exp(-dt / 360.0), 0.0, 1.0)
            self._apply_ground_mask()
        self._pending_smoke.fill(0.0)
        self._pending_flame.fill(0.0)
        self._pending_temperature.fill(self.ambient_temp_k)

    def snapshot(self) -> Dict[str, object]:
        """Retourne une representation compacte pour texture 3D cote navigateur."""
        temp = np.clip((self.state.temperature_k - self.ambient_temp_k) / 1300.0, 0.0, 1.0)
        return {
            "shape": [self.d, self.h, self.w],
            "domain_m": [self.domain_x_m, self.domain_y_m, self.domain_z_m],
            "smoke": np.rint(self.state.smoke_density * 255.0).astype(np.uint8).flatten().tolist(),
            "cloud": np.rint(self.state.cloud_water * 255.0).astype(np.uint8).flatten().tolist(),
            "flame": np.rint(self.state.flame_density * 255.0).astype(np.uint8).flatten().tolist(),
            "temperature": np.rint(temp * 255.0).astype(np.uint8).flatten().tolist(),
            "relative_humidity": np.rint(np.clip(self.state.relative_humidity_pct, 0.0, 100.0) * 2.55).astype(np.uint8).flatten().tolist(),
            "precipitation": np.rint(np.clip(self.state.precipitation_mm_h, 0.0, 100.0) * 2.55).astype(np.uint8).flatten().tolist(),
            "wind_mean_ms": [
                round(float(np.mean(self.wind_u)), 2),
                round(float(np.mean(self.wind_v)), 2),
                round(float(np.mean(self.wind_w)), 2),
            ],
            "weather_field": "spatial_surface_interpolated_vertical_profile",
            "terrain_following": True,
            "ground_height_m": np.rint(self.ground_height_m).astype(np.uint16).flatten().tolist(),
            "boundary_m": [-self.domain_x_m / 2.0, self.domain_x_m / 2.0, 0.0, self.domain_y_m, -self.domain_z_m / 2.0, self.domain_z_m / 2.0],
        }
