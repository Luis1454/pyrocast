import numpy as np

from src.weather.live_weather_service import LiveWeatherService


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_fetch_spatial_weather_grid_uses_multiple_real_observations(monkeypatch):
    records = []
    for index in range(9):
        records.append({
            "current": {
                "temperature_2m": 20.0 + index,
                "relative_humidity_2m": 45.0 + index,
                "surface_pressure": 1000.0 + index,
                "wind_speed_10m": 10.0 + index,
                "wind_direction_10m": 270.0,
                "wind_gusts_10m": 15.0 + index,
                "precipitation": 0.1 * index,
                "cloud_cover": 30.0 + index,
            }
        })
    monkeypatch.setattr(
        "src.weather.live_weather_service.requests.get",
        lambda *args, **kwargs: _Response(records),
    )

    service = LiveWeatherService()
    field = service.fetch_spatial_weather_grid(43.5, 5.4, 6400.0, sample_grid=3)

    assert field["is_live_api"] is True
    assert field["sample_grid"] == 3
    assert field["temperature_c_grid"][0][0] != field["temperature_c_grid"][-1][-1]
    assert field["wind_u_grid"][0][0] > 0.0

    elev = np.zeros((8, 8), dtype=np.float32)
    micro = service.compute_spatial_microclimate_grid(elev, 50.0, {"spatial_weather_grid": field, "temperature_c": 20.0, "relative_humidity_pct": 45.0, "wind_speed_kmh": 10.0, "wind_direction_deg": 270.0, "precipitation_mm_h": 0.0})
    assert micro["weather_field_is_live"] is True
    assert micro["temperature_grid"][0][0] != micro["temperature_grid"][-1][-1]


def test_wind_field_is_tangent_to_relief_and_exposes_terrain_geometry():
    service = LiveWeatherService()
    elevation = np.tile(np.linspace(0.0, 400.0, 12, dtype=np.float32), (12, 1))
    surface = np.ones((3, 3), dtype=np.float32)
    spatial = {
        "is_live_api": True,
        "source": "test",
        "temperature_c_grid": (25.0 * surface).tolist(),
        "relative_humidity_pct_grid": (45.0 * surface).tolist(),
        "precipitation_mm_h_grid": np.zeros_like(surface).tolist(),
        "cloud_cover_pct_grid": (20.0 * surface).tolist(),
        "wind_u_grid": (10.0 * surface).tolist(),
        "wind_v_grid": np.zeros_like(surface).tolist(),
    }
    micro = service.compute_spatial_microclimate_grid(
        elevation,
        50.0,
        {
            "spatial_weather_grid": spatial,
            "temperature_c": 25.0,
            "relative_humidity_pct": 45.0,
            "wind_speed_kmh": 36.0,
            "wind_direction_deg": 270.0,
            "precipitation_mm_h": 0.0,
        },
    )

    wind_w = np.asarray(micro["wind_w_grid"], dtype=np.float32)
    terrain_gradient = np.asarray(micro["terrain_gradient_x_grid"], dtype=np.float32)
    assert float(terrain_gradient.mean()) > 0.1
    assert float(wind_w.mean()) > 0.5
    assert np.asarray(micro["terrain_height_render_m_grid"]).max() > 0.0
    assert micro["wind_field_model"].startswith("terrain_tangent_surface")
