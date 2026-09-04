import numpy as np

from src.physics.atmospheric_volume import AtmosphericVolumeSolver


def test_atmospheric_scalars_share_one_wind_field():
    solver = AtmosphericVolumeSolver(shape=(12, 24, 24))
    solver.set_wind(wind_u_ms=8.0, wind_v_ms=-3.0)
    solver.initialize_cloud_layer(cloud_cover_pct=80.0, relative_humidity_pct=75.0)
    solver.inject_fire_sources([(0.0, 0.0, 0.0, 24.0, 1.0)])

    solver.step(dt_s=12.0, substeps=3)
    snapshot = solver.snapshot()

    assert np.isfinite(solver.state.smoke_density).all()
    assert np.isfinite(solver.state.cloud_water).all()
    assert float(solver.state.smoke_density.max()) > 0.0
    assert float(solver.state.cloud_water.max()) > 0.0
    assert snapshot["shape"] == [12, 24, 24]
    assert len(snapshot["smoke"]) == 12 * 24 * 24
    assert snapshot["wind_mean_ms"][0] > 0.0
    assert snapshot["wind_mean_ms"][1] < 0.0


def test_spatial_weather_is_projected_into_volume_and_rain_scavenges_smoke():
    shape = (8, 12, 12)
    surface = np.ones((4, 4), dtype=np.float32)
    microclimate = {
        "temperature_grid": (24.0 + surface * 2.0).tolist(),
        "relative_humidity_grid": (55.0 + surface * 10.0).tolist(),
        "precipitation_mm_h_grid": (12.0 * surface).tolist(),
        "cloud_cover_pct_grid": (70.0 * surface).tolist(),
        "wind_u_grid": (surface * 6.0).tolist(),
        "wind_v_grid": (surface * -2.0).tolist(),
        "wind_w_grid": np.zeros_like(surface).tolist(),
    }
    solver = AtmosphericVolumeSolver(shape=shape)
    solver.set_meteorological_fields(microclimate)
    solver.inject_fire_sources([(0.0, 0.0, 0.0, 30.0, 1.0)])
    solver.step(dt_s=30.0, substeps=3)

    dry_solver = AtmosphericVolumeSolver(shape=shape)
    dry_microclimate = dict(microclimate, precipitation_mm_h_grid=np.zeros_like(surface).tolist())
    dry_solver.set_meteorological_fields(dry_microclimate)
    dry_solver.inject_fire_sources([(0.0, 0.0, 0.0, 30.0, 1.0)])
    dry_solver.step(dt_s=30.0, substeps=3)

    assert solver.state.temperature_k[0].mean() > 273.15
    assert float(solver.state.relative_humidity_pct.mean()) > 50.0
    assert float(solver.state.precipitation_mm_h.mean()) > 0.0
    assert float(solver.state.smoke_density.max()) < float(dry_solver.state.smoke_density.max())
    assert solver.snapshot()["boundary_m"] == [-3200.0, 3200.0, 0.0, 2400.0, -3200.0, 3200.0]


def test_volume_masks_subterrain_cells_when_mnt_is_provided():
    terrain = np.full((4, 4), 180.0, dtype=np.float32)
    microclimate = {
        "temperature_grid": np.full((4, 4), 25.0).tolist(),
        "relative_humidity_grid": np.full((4, 4), 60.0).tolist(),
        "precipitation_mm_h_grid": np.zeros((4, 4)).tolist(),
        "cloud_cover_pct_grid": np.full((4, 4), 50.0).tolist(),
        "wind_u_grid": np.full((4, 4), 5.0).tolist(),
        "wind_v_grid": np.zeros((4, 4)).tolist(),
        "wind_w_grid": np.zeros((4, 4)).tolist(),
        "terrain_height_render_m_grid": terrain.tolist(),
    }
    solver = AtmosphericVolumeSolver(shape=(8, 8, 8), domain_size_m=(6400.0, 900.0, 6400.0))
    solver.set_meteorological_fields(microclimate)
    solver.inject_fire_sources([(0.0, 180.0, 0.0, 30.0, 1.0)])

    assert np.all(solver.state.smoke_density[0:2] == 0.0)
    assert np.all(solver.state.cloud_water[0:2] == 0.0)
    assert np.all(solver.wind_u[0:2] == 0.0)
    assert solver.snapshot()["terrain_following"] is True
