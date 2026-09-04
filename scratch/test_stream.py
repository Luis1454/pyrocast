import time
from src.physics.fire_engineering import StochasticMonteCarloSpreadSimulator
import numpy as np

t0 = time.time()
elev = 300.0 + np.random.normal(0, 10, (128, 128)).astype(np.float32)
sim = StochasticMonteCarloSpreadSimulator(elev, dx_meters=50.0, fuel_code="SH5")
def cb(fr):
    pass
res = sim.simulate_ensemble(
    ignition_point_px=(64, 56),
    wind_speed_kmh=45.0,
    wind_dir_cardinal="NW",
    fmc_pct=5.0,
    num_ensembles=10,
    num_output_steps=30,
    frame_callback=cb
)
print(f"30 frames on 128x128 grid completed in {time.time()-t0:.2f}s")
