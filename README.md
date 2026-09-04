# PyroCast

AI-powered wildfire spread prediction and operational simulation.

PyroCast combines wildfire physics, weather forcing, terrain-aware meshes,
Monte Carlo inference, 2D/3D visualization, and operational exports for
field-mission analysis.

## Visual overview

<p align="center">
  <img src="reports/playwright/01_firemap_pro_3d_c2_view.png" alt="PyroCast 3D atmospheric view" width="48%">
  <img src="reports/playwright/02_firemap_pro_2d_sig_view.png" alt="PyroCast 2D tactical GIS view" width="48%">
</p>

<p align="center">
  <img src="reports/playwright/04_firemap_pro_thermodynamic_flows_view.png" alt="PyroCast thermodynamic flows" width="48%">
  <img src="reports/field_mission/05_tactical_operational_map.png" alt="PyroCast tactical fire-risk map" width="48%">
</p>

## Main entry points

```bash
python app.py
python firemap_cli.py --help
python run_field_mission.py
python run_3d_cfd_simulation.py
python infer.py
```

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Optional graph experiments use `requirements-optional-graph.txt`.
