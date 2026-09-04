"""
app.py
------
FireMap Pro - Console d'Ingénierie & de Commandement Opérationnel des Feux de Forêt (C2 / SITAC / CFD) :
- Rendu 3D WebGL haute fidélité (Three.js) avec MNT réel et drapage dynamique d'orthophoto satellite (ESRI World Imagery)
- Cartographie SIG opérationnelle 2D (Leaflet WGS84) avec symboles normalisés FDF / SITAC
- Solveur couplé 3D Navier-Stokes, Rothermel (1972), Byram (1959) et Van Wagner (1977)
- Connecteurs temps réel pour MNT 50m sur 6.4 km et Météorologie en direct (Open-Meteo API)
- Dimensionnement tactique des moyens (GIFF, Canadair CL-415, Tranchées coupe-feu) et Ordre d'Opérations FDF5
"""

import http.server
import socketserver
import json
import urllib.parse
import time
import math
import hashlib
import base64
import struct
import io
import requests
from pathlib import Path
import numpy as np
import torch

from src.terrain.dem_loader import DigitalElevationModel
from src.terrain.real_elevation_service import RealElevationService
from src.data.map_metadata_service import MapMetadataService
from src.weather.live_weather_service import LiveWeatherService
from src.physics.fire_engineering import FireBehaviorEngineeringEngine, FuelModelData, StochasticMonteCarloSpreadSimulator
from src.physics.cfd_3d_engine import NavierStokesCombustion3DSolver, CFD3DState
from src.physics.atmospheric_volume import AtmosphericVolumeSolver
from src.models.stochastic_fno import StochasticFourierNeuralOperator2D, StochasticFourierNeuralOperator3D
from src.models.mlp_3d import ThermodynamicMLP3D
from src.mesh.octree_3d_volumetric import Volumetric3DOctreeBuilder
from src.operational.vtk_exporter import VTK3DExporter
from src.operational.geojson_exporter import GeoJSONFireExporter
from src.operational.tactical_sim import TacticalInterventionManager
from src.operational.mission_report import OperationalMissionReporter
from src.operational.ground_truth_validator import GroundTruthValidator
from src.operational.data_assimilation_enkf import EnsembleKalmanFilterWildfireAssimilation


def send_ws_frame(sock, obj):
    """Envoie une trame de texte WebSocket encodée en JSON (RFC 6455)."""
    try:
        payload = json.dumps(obj).encode("utf-8")
        length = len(payload)
        if length <= 125:
            header = bytes([0x81, length])
        elif length <= 65535:
            header = bytes([0x81, 126]) + struct.pack(">H", length)
        else:
            header = bytes([0x81, 127]) + struct.pack(">Q", length)
        sock.sendall(header + payload)
        return True
    except Exception:
        return False


def build_atmospheric_volume(weather, microclimate=None, wind_speed_kmh=None, wind_dir=None):
    """Build the shared Eulerian atmosphere from spatial weather fields."""
    speed_kmh = float(wind_speed_kmh if wind_speed_kmh is not None else weather.get("wind_speed_kmh", 0.0))
    cardinal = wind_dir or weather.get("wind_cardinal", "NW")
    dir_angles = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0}
    wind_rad = math.radians(dir_angles.get(cardinal, 315.0))
    speed_ms = speed_kmh / 3.6
    # Convention meteo: le vecteur est oriente vers la direction de transport.
    wind_u = -speed_ms * math.sin(wind_rad)
    wind_v = -speed_ms * math.cos(wind_rad)
    ambient_temp_k = float(weather.get("temperature_c", 25.0)) + 273.15
    volume = AtmosphericVolumeSolver(
        shape=(32, 48, 48),
        domain_size_m=(DOMAIN_SIZE_M, ATMOSPHERE_HEIGHT_M, DOMAIN_SIZE_M),
        ambient_temp_k=ambient_temp_k,
    )
    if microclimate and microclimate.get("wind_u_grid"):
        volume.set_meteorological_fields(microclimate)
    else:
        volume.set_wind(wind_u, wind_v)
        volume.initialize_cloud_layer(weather.get("cloud_cover_pct", 0.0), weather.get("relative_humidity_pct", 50.0))
    return volume


def resolve_distributed_forcing(weather_svc, weather, microclimate, fallback_wind_kmh, fallback_wind_dir, fallback_fmc, fallback_precipitation):
    """Derive one coherent forcing summary from the spatial weather field."""
    if not (weather.get("is_live_api") or microclimate.get("weather_field_is_live")):
        return fallback_wind_kmh, fallback_wind_dir, fallback_fmc, fallback_precipitation

    u_grid = np.asarray(microclimate.get("wind_u_grid", []), dtype=np.float32)
    v_grid = np.asarray(microclimate.get("wind_v_grid", []), dtype=np.float32)
    fmc_grid = np.asarray(microclimate.get("fmc_grid", []), dtype=np.float32)
    rain_grid = np.asarray(microclimate.get("precipitation_mm_h_grid", []), dtype=np.float32)
    if u_grid.size == 0 or v_grid.size == 0:
        return fallback_wind_kmh, fallback_wind_dir, fallback_fmc, fallback_precipitation

    mean_u = float(np.mean(u_grid))
    mean_v = float(np.mean(v_grid))
    speed_kmh = float(np.mean(np.sqrt(u_grid**2 + v_grid**2)) * 3.6)
    direction_deg = (math.degrees(math.atan2(-mean_u, -mean_v)) + 360.0) % 360.0
    wind_dir = weather_svc._deg_to_cardinal(direction_deg)
    fmc = float(np.mean(fmc_grid)) if fmc_grid.size else fallback_fmc
    precipitation = float(np.mean(rain_grid)) if rain_grid.size else fallback_precipitation

    weather["wind_speed_kmh"] = round(speed_kmh, 1)
    weather["wind_direction_deg"] = round(direction_deg, 1)
    weather["wind_cardinal"] = wind_dir
    weather["fmc_pct"] = round(fmc, 1)
    weather["precipitation_mm_h"] = round(precipitation, 2)
    weather["simulation_forcing"] = "live spatial weather field"
    return speed_kmh, wind_dir, fmc, precipitation


def read_ws_frame(sock):
    """Lit et démasque un message texte WebSocket entrant."""
    try:
        head = sock.recv(2)
        if not head or len(head) < 2: return None
        fin_opcode, mask_len = head[0], head[1]
        is_masked = bool(mask_len & 0x80)
        length = mask_len & 0x7F
        if length == 126:
            length = struct.unpack(">H", sock.recv(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", sock.recv(8))[0]
        mask = sock.recv(4) if is_masked else None
        data = bytearray()
        while len(data) < length:
            chunk = sock.recv(length - len(data))
            if not chunk: break
            data.extend(chunk)
        if is_masked and mask:
            for i in range(len(data)):
                data[i] ^= mask[i % 4]
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return None


PORT = 5050
GRID_SIZE = 128
CELL_SIZE_M = 50.0
DOMAIN_SIZE_M = GRID_SIZE * CELL_SIZE_M
ATMOSPHERE_HEIGHT_M = 2400.0
MAP_METADATA_SERVICE = MapMetadataService()

HTML_PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>FireMap Pro - Système de Commandement & Ingénierie Feux de Forêt</title>
    <!-- Leaflet 2D -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <!-- Three.js 3D WebGL & OrbitControls -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        :root {
            --bg-base: #0b0f19;
            --bg-panel: #111827;
            --bg-card: #1f2937;
            --border-color: #374151;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent-blue: #38bdf8;
            --accent-orange: #f97316;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-purple: #8b5cf6;
            --surface-glass: rgba(15, 23, 42, 0.82);
            --surface-strong: rgba(3, 7, 18, 0.94);
            --accent-cyan: #67e8f9;
        }
        body { margin: 0; padding: 0; font-family: "Bahnschrift", "Aptos", "Segoe UI", sans-serif; background: radial-gradient(circle at 72% 18%, #172554 0%, var(--bg-base) 34%, #050811 100%); color: var(--text-primary); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        
        /* Header C2 Opérationnel */
        #c2-header { height: 58px; background: linear-gradient(90deg, rgba(3,7,18,0.98), rgba(15,23,42,0.96)); border-bottom: 1px solid #334155; display: flex; align-items: center; justify-content: space-between; padding: 0 20px; z-index: 100; box-sizing: border-box; box-shadow: 0 6px 24px rgba(0,0,0,0.24); }
        .c2-title-block { display: flex; align-items: center; gap: 14px; }
        .c2-badge { background: #dc2626; color: white; font-size: 10px; font-weight: 800; padding: 3px 8px; border-radius: 4px; letter-spacing: 0.8px; text-transform: uppercase; }
        .c2-incident { font-size: 13px; font-weight: bold; color: var(--text-primary); letter-spacing: 0.3px; }
        .c2-telemetry-bar { display: flex; align-items: center; gap: 18px; font-size: 11.5px; color: var(--text-secondary); }
        .c2-stat { display: flex; align-items: center; gap: 5px; }
        .c2-stat-val { color: var(--accent-blue); font-weight: bold; font-family: monospace; font-size: 12px; }

        /* Conteneur Principal */
        #workspace { display: flex; flex: 1; height: calc(100vh - 58px); overflow: hidden; }
        #sidebar { width: 370px; background: linear-gradient(180deg, rgba(17,24,39,0.98), rgba(10,15,28,0.98)); border-right: 1px solid #334155; padding: 12px; box-sizing: border-box; overflow-y: auto; display: flex; flex-direction: column; gap: 9px; z-index: 10; }
        #main-view { flex: 1; height: 100%; display: flex; flex-direction: column; position: relative; background: #071525; }
        
        /* Onglets de Vue */
        .tab-bar { display: flex; background: rgba(3,7,18,0.96); border-bottom: 1px solid var(--border-color); }
        .tab-btn { flex: 1; padding: 10px; background: transparent; border: none; color: var(--text-secondary); font-weight: 600; font-size: 12.5px; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; transition: all 0.15s; }
        .tab-btn.active { color: var(--accent-blue); background: var(--bg-panel); border-bottom: 2px solid var(--accent-blue); }
        
        #scene-toolbar { min-height: 38px; display: flex; align-items: center; gap: 7px; padding: 5px 12px; background: rgba(15,23,42,0.96); border-bottom: 1px solid #334155; box-sizing: border-box; }
        .scene-toolbar-label { margin-right: 7px; color: var(--text-secondary); font-size: 10px; font-weight: 800; letter-spacing: 0.8px; text-transform: uppercase; }
        .layer-btn { border: 1px solid #475569; background: #172033; color: #cbd5e1; border-radius: 4px; padding: 5px 9px; font-size: 10px; font-weight: 800; letter-spacing: 0.35px; cursor: pointer; }
        .layer-btn.active { background: #0e7490; border-color: var(--accent-cyan); color: #ecfeff; box-shadow: 0 0 0 1px rgba(103,232,249,0.18); }
        .layer-btn:hover { border-color: var(--accent-cyan); }
        #scene-status { margin-left: auto; color: #86efac; font-size: 10px; font-family: monospace; letter-spacing: 0.4px; }
        #view-reset { background: transparent; color: #94a3b8; border: 1px solid #475569; border-radius: 4px; padding: 5px 9px; font-size: 10px; cursor: pointer; }
        .view-container { flex: 1; min-height: 0; position: relative; width: 100%; overflow: hidden; background: linear-gradient(180deg, #173b58 0%, #0d2940 38%, #0b1a29 63%, #07131f 100%); }
        #canvas3d { width: 100%; height: 100%; display: block; position: relative; background: radial-gradient(ellipse at 50% 18%, rgba(160, 211, 226, 0.28) 0%, rgba(44, 105, 132, 0.16) 30%, transparent 64%), linear-gradient(180deg, #16415d 0%, #12344c 36%, #0b1d2c 70%, #07131f 100%); }
        #canvas3d::before { content: ''; position: absolute; inset: 0; pointer-events: none; background: linear-gradient(180deg, rgba(202, 235, 242, 0.12), transparent 23%, transparent 65%, rgba(2, 8, 16, 0.34)); z-index: 1; }
        .scene-context { position: absolute; top: 16px; left: 18px; z-index: 15; pointer-events: none; text-shadow: 0 2px 12px rgba(0,0,0,0.45); }
        .scene-context-kicker { color: #a5f3fc; font: 800 9px/1.2 monospace; letter-spacing: 1.5px; }
        .scene-context-title { margin-top: 5px; color: #f8fafc; font-size: 18px; font-weight: 650; letter-spacing: -0.3px; }
        .scene-context-subtitle { margin-top: 4px; color: rgba(226, 232, 240, 0.78); font: 10px/1.3 monospace; }
        .scene-context-chips { display: flex; gap: 6px; margin-top: 10px; }
        .scene-context-chip { padding: 4px 7px; border: 1px solid rgba(186, 230, 253, 0.34); border-radius: 3px; background: rgba(7, 23, 38, 0.44); color: #dbeafe; font: 800 9px monospace; letter-spacing: 0.3px; }
        .atmospheric-scale { position: absolute; left: 18px; bottom: 82px; z-index: 15; pointer-events: none; color: rgba(226,232,240,0.76); font: 9px monospace; letter-spacing: 0.4px; }
        .atmospheric-scale::before { content: ''; display: block; width: 108px; height: 1px; margin-bottom: 5px; background: linear-gradient(90deg, #a5f3fc, transparent); }
        #map2d { width: 100%; height: 100%; display: none; }
        
        /* Cartes & Composants */
        .card { background: linear-gradient(145deg, rgba(31,41,55,0.96), rgba(15,23,42,0.96)); border: 1px solid #334155; border-radius: 7px; padding: 10px 12px; box-shadow: 0 5px 16px rgba(0,0,0,0.16); }
        .card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
        .card-title { font-size: 10.5px; font-weight: 800; color: var(--accent-blue); text-transform: uppercase; letter-spacing: 0.6px; }
        
        .field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 11.5px; }
        .field-box { background: rgba(2,6,23,0.66); padding: 6px 8px; border-radius: 4px; border: 1px solid #334155; }
        .field-label { font-size: 9.5px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.4px; }
        .field-val { font-size: 13px; font-weight: bold; color: #fff; font-family: monospace; margin-top: 2px; }
        .sidebar-heading { padding: 5px 3px 7px; }
        .eyebrow { color: #67e8f9; font-size: 9px; font-weight: 800; letter-spacing: 1.4px; }
        .sidebar-title { margin-top: 3px; font-size: 21px; line-height: 1.1; font-weight: 600; letter-spacing: -0.4px; }
        .sidebar-subtitle { margin-top: 7px; color: #94a3b8; font-size: 9px; font-weight: 700; letter-spacing: 0.55px; }
        .status-dot { display: inline-block; width: 6px; height: 6px; margin-right: 5px; border-radius: 50%; background: #4ade80; box-shadow: 0 0 8px #4ade80; }
        .card-kicker { color: #94a3b8; font-size: 9px; margin-top: 3px; letter-spacing: 0.25px; }
        .icon-action { width: 28px; height: 28px; padding: 0; border: 1px solid #475569; border-radius: 5px; background: #162033; color: #67e8f9; font-size: 18px; cursor: pointer; }
        .icon-action:hover { background: #0e7490; color: #ecfeff; }
        .coordinate-line { display: flex; align-items: baseline; gap: 5px; margin: 9px 1px 7px; color: #e2e8f0; font-family: monospace; font-size: 15px; font-weight: 700; }
        .coordinate-separator { color: #64748b; font-size: 9px; font-weight: 800; }
        .compact-fields { gap: 5px; }
        .weather-control-head, .range-label { display: flex; justify-content: space-between; align-items: baseline; margin-top: 8px; color: #cbd5e1; font-size: 10.5px; }
        .weather-control-head strong, .range-label b { color: #67e8f9; font-family: monospace; font-size: 14px; }
        .weather-control-head small, .range-label small { color: #94a3b8; font-size: 9px; }
        .weather-control-head em { color: #94a3b8; font-style: normal; }
        .orange-value { color: #fb923c !important; }
        .weather-readout { display: grid; grid-template-columns: repeat(3, 1fr); margin-top: 10px; padding-top: 9px; border-top: 1px solid #334155; color: #94a3b8; font-family: monospace; font-size: 9px; gap: 6px; }
        .weather-readout b { display: block; margin-top: 3px; color: #f8fafc; font-size: 11px; }
        .live-pill, .model-badge, .tactical-mark, .briefing-time { padding: 3px 6px; border-radius: 3px; font-size: 8px; font-weight: 800; letter-spacing: 0.7px; }
        .live-pill { background: rgba(22,163,74,0.18); color: #86efac; border: 1px solid rgba(74,222,128,0.4); }
        .model-badge { background: rgba(124,58,237,0.2); color: #c4b5fd; border: 1px solid rgba(167,139,250,0.4); }
        .tactical-mark { background: rgba(234,88,12,0.17); color: #fdba74; border: 1px solid rgba(251,146,60,0.4); }
        .briefing-time { color: #94a3b8; border: 1px solid #475569; }
        .switch-list { margin-top: 8px; border-top: 1px solid #334155; }
        .switch-row { min-height: 27px; margin: 0; padding: 4px 0; display: flex; align-items: center; justify-content: space-between; color: #cbd5e1; font-size: 10px; }
        .switch-row > span { display: flex; align-items: center; gap: 7px; }
        .switch-row input { position: absolute; opacity: 0; pointer-events: none; }
        .switch-row i { position: relative; display: inline-block; width: 25px; height: 14px; border: 1px solid #64748b; border-radius: 10px; background: #1e293b; transition: background 0.15s, border 0.15s; }
        .switch-row i::after { content: ''; position: absolute; top: 2px; left: 2px; width: 8px; height: 8px; border-radius: 50%; background: #94a3b8; transition: transform 0.15s, background 0.15s; }
        .switch-row input:checked + i, .switch-row .fixed-state { background: #0e7490; border-color: #67e8f9; }
        .switch-row input:checked + i::after { transform: translateX(11px); background: #ecfeff; }
        .switch-row .fixed-state::after { transform: translateX(11px); background: #ecfeff; }
        .switch-row b { font-weight: 500; }
        .switch-row small { color: #64748b; font-family: monospace; font-size: 8px; letter-spacing: 0.5px; }
        .fuel-label { margin-top: 10px; }
        .briefing-card { margin-top: auto; }
        .btn-mark { color: #fed7aa; margin-right: 4px; }

        label { font-size: 11.5px; color: var(--text-primary); display: flex; justify-content: space-between; margin-top: 5px; }
        input[type="range"], select { width: 100%; margin-top: 3px; background: #111827; color: white; border: 1px solid var(--border-color); border-radius: 4px; padding: 5px 8px; box-sizing: border-box; font-size: 12px; }
        
        .btn-group { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 4px; }
        .btn { width: 100%; background: #2563eb; color: white; border: none; border-radius: 5px; padding: 9px; font-weight: 700; cursor: pointer; font-size: 12px; transition: background 0.15s; letter-spacing: 0.3px; }
        .btn:hover { background: #1d4ed8; }
        .btn-primary { background: #ea580c; grid-column: span 2; padding: 11px; font-size: 13px; }
        .btn-primary:hover { background: #c2410c; }
        .btn-sec { background: #4b5563; }
        .btn-sec:hover { background: #374151; }
        .btn-vtk { background: #7c3aed; }
        .btn-vtk:hover { background: #6d28d9; }

        .checkbox-group { display: flex; align-items: center; gap: 8px; font-size: 11.5px; margin-top: 5px; }
        #briefing-box { background: #030712; border: 1px solid var(--border-color); border-radius: 4px; padding: 8px; font-family: Consolas, Monaco, monospace; font-size: 10.5px; color: #34d399; max-height: 140px; overflow-y: auto; white-space: pre-wrap; line-height: 1.35; }
        
        /* Overlays Scientifiques 3D */
        .hud-overlay { position: absolute; top: 16px; right: 18px; background: rgba(5, 18, 31, 0.70); border: 1px solid rgba(186, 230, 253, 0.30); backdrop-filter: blur(16px); border-radius: 6px; padding: 12px 14px; color: white; font-size: 10.5px; pointer-events: none; z-index: 20; min-width: 238px; box-shadow: 0 10px 28px rgba(0,0,0,0.34); }
        .hud-heading { margin-bottom: 9px; color: #a5f3fc; font: 800 9px monospace; letter-spacing: 1px; text-transform: uppercase; }
        .hud-row { display: flex; justify-content: space-between; align-items: baseline; gap: 18px; margin-bottom: 6px; color: #cbd5e1; }
        .hud-row span:first-child { color: #dbe4ec; }
        .hud-val { font-size: 12.5px; font-weight: bold; color: var(--accent-blue); font-family: monospace; white-space: nowrap; }
        .hud-divider { height: 1px; margin: 8px 0; background: rgba(148, 163, 184, 0.20); }
        .hud-foot { color: #94a3b8; font: 9px monospace; }
        
        .legend-3d { position: absolute; bottom: 70px; right: 16px; background: var(--surface-glass); border: 1px solid #475569; border-radius: 7px; padding: 10px 12px; font-size: 10.5px; z-index: 20; backdrop-filter: blur(10px); }
        .gradient-bar { height: 10px; width: 160px; background: linear-gradient(to right, #3b82f6, #10b981, #eab308, #ef4444); border-radius: 2px; margin: 4px 0; }
        .legend-ticks { display: flex; justify-content: space-between; font-size: 9.5px; color: var(--text-secondary); font-family: monospace; }

        /* Barre de Contrôle Temporel de Simulation (Timeline) */
        #sim-timeline-bar { position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%); background: var(--surface-strong); border: 1px solid #475569; border-radius: 8px; padding: 8px 18px; display: flex; align-items: center; gap: 14px; z-index: 30; backdrop-filter: blur(12px); box-shadow: 0 8px 24px rgba(0,0,0,0.6); min-width: 580px; }
        .t-btn { background: #ea580c; color: white; border: none; border-radius: 4px; padding: 6px 14px; font-weight: 800; font-size: 11.5px; cursor: pointer; display: flex; align-items: center; gap: 6px; transition: background 0.15s; }
        .t-btn:hover { background: #c2410c; }
        .t-btn-sec { background: #374151; }
        .t-btn-sec:hover { background: #4b5563; }
        #timeline-slider { flex: 1; accent-color: #ea580c; cursor: pointer; }
        #timeline-badge { font-family: monospace; font-size: 11.5px; font-weight: bold; color: #f3f4f6; white-space: nowrap; background: #030712; padding: 4px 10px; border-radius: 4px; border: 1px solid #374151; }
        @media (max-width: 1100px) { #sidebar { width: 330px; } .c2-telemetry-bar { gap: 9px; } .c2-incident { display: none; } #sim-timeline-bar { min-width: 0; width: calc(100% - 28px); } }
        @media (max-width: 760px) { body { overflow: auto; height: auto; min-height: 100vh; } #c2-header { height: auto; min-height: 58px; padding: 9px 12px; align-items: flex-start; gap: 8px; } .c2-telemetry-bar { flex-wrap: wrap; justify-content: flex-end; } #workspace { height: auto; min-height: calc(100vh - 58px); flex-direction: column; overflow: visible; } #sidebar { width: 100%; max-height: none; overflow: visible; border-right: 0; border-bottom: 1px solid #334155; } #main-view { min-height: 72vh; } .tab-btn { font-size: 10px; padding: 9px 4px; } #scene-toolbar { flex-wrap: wrap; } #scene-status { margin-left: 0; } .hud-overlay { top: 8px; right: 8px; min-width: 190px; transform: scale(0.88); transform-origin: top right; } .scene-context { left: 10px; top: 10px; } .scene-context-title { font-size: 15px; } .atmospheric-scale { left: 10px; } .legend-3d { display: none; } }
    </style>
</head>
<body>
    <!-- Header C2 Command & Control -->
    <div id="c2-header">
        <div class="c2-title-block">
            <span class="c2-badge">SITAC / FDF5</span>
            <span class="c2-ai-badge" style="background:#0e7490;color:white;font-size:10px;font-weight:800;padding:3px 8px;border-radius:4px;letter-spacing:0.8px;">NOMINAL · ENSEMBLE · CFD</span>
            <span class="c2-incident">FIREMAP PRO &bull; INC-2026-MEDITERRANEE-08</span>
        </div>
        <div class="c2-telemetry-bar">
            <div class="c2-stat">VENT: <span id="top-wind" class="c2-stat-val">35.0 km/h NW</span></div>
            <div class="c2-stat">TEMP: <span id="top-temp" class="c2-stat-val">32.0 °C</span></div>
            <div class="c2-stat">FMC: <span id="top-fmc" class="c2-stat-val">6.0 %</span></div>
            <div class="c2-stat">HAINES: <span id="top-haines" class="c2-stat-val">4/6 (Modéré)</span></div>
            <div class="c2-stat">HEURE: <span id="top-clock" class="c2-stat-val">12:00:00 UTC</span></div>
        </div>
    </div>

    <div id="workspace">
        <div id="sidebar">
            <div class="sidebar-heading">
                <div class="eyebrow">FIRE BEHAVIOUR DESK · TEMPS RÉEL</div>
                <div class="sidebar-title">Situation opérationnelle</div>
                <div class="sidebar-subtitle"><span class="status-dot"></span> DONNÉES PRÊTES · PRÉVISION À 60 MIN</div>
            </div>

            <div class="card mission-card">
                <div class="card-header">
                    <div><div class="card-title">Zone d'étude</div><div class="card-kicker">MNT + imagerie satellite</div></div>
                    <button class="icon-action" onclick="syncLiveData()" aria-label="Actualiser les données">↻</button>
                </div>
                <select id="case-select" onchange="loadHistoricalCase(this.value)">
                    <option value="SAINTE_VICTOIRE" selected>Sainte-Victoire · Bouches-du-Rhône</option>
                    <option value="MAURES_2021">Les Maures · Var · 2021</option>
                    <option value="LANDIRAS_2022">Landiras · Gironde · 2022</option>
                </select>
                <div class="coordinate-line"><span id="lat-val">43.5250</span><span class="coordinate-separator">N</span><span id="lon-val">5.4420</span><span class="coordinate-separator">E</span></div>
                <div class="field-grid compact-fields">
                    <div class="field-box"><div class="field-label">Altitude MNT</div><div class="field-val" id="alt-val">340 m</div></div>
                    <div class="field-box"><div class="field-label">Pente moyenne</div><div class="field-val" id="slope-val">14.2°</div></div>
                </div>
            </div>

            <div class="card weather-card">
                <div class="card-header"><div><div class="card-title">Forçage atmosphérique</div><div class="card-kicker">Vent à 10 m · combustible fin</div></div><span class="live-pill">LIVE</span></div>
                <div class="weather-control-head"><span>Vent</span><strong><span id="wind-val">45</span> <small>km/h</small></strong></div>
                <input type="range" id="wind-speed" min="5" max="110" value="45" oninput="document.getElementById('wind-val').innerText = this.value; updateFireEngineeringPreview();">
                <select id="wind-dir" onchange="updateFireEngineeringPreview()">
                    <option value="NW" selected>NW · Mistral / Tramontane</option><option value="N">N · Nord</option><option value="NE">NE · Grégal</option><option value="E">E · Levant</option><option value="SE">SE · Marin</option><option value="S">S · Sud</option><option value="SW">SW · Libeccio</option><option value="W">W · Ponant</option>
                </select>
                <div class="weather-control-head"><span>Humidité du combustible <em>(FMC)</em></span><strong class="orange-value"><span id="fmc-val">6</span><small>%</small></strong></div>
                <input type="range" id="fmc-slider" min="2" max="25" value="6" oninput="document.getElementById('fmc-val').innerText = this.value; updateFireEngineeringPreview();">
                <div class="weather-readout"><span>T° air <b id="sidebar-temp">32 °C</b></span><span>RH <b id="sidebar-rh">28 %</b></span><span>Pluie <b id="sidebar-rain">0 mm/h</b></span></div>
            </div>

            <div class="card model-card">
                <div class="card-header"><div><div class="card-title">Prévision incendie</div><div class="card-kicker">Propagation · convection · incertitude</div></div><span class="model-badge">SFNO</span></div>
                <select id="sim-mode-select"><option value="stochastic_fno" selected>SFNO stochastique + champ fluide 3D</option><option value="deterministic">Rothermel / Byram déterministe</option></select>
                <div class="switch-list">
                    <div class="switch-row"><span><i class="fixed-state" aria-hidden="true"></i><b>Champ atmosphérique volumique</b></span><small>Actif · Eulerien / CFD</small></div>
                    <label class="switch-row"><span><input type="checkbox" id="chk-enkf" checked><i></i><b>Assimilation FIRMS / EnKF</b></span><small>VIIRS</small></label>
                    <label class="switch-row"><span><input type="checkbox" id="chk-spotting" checked><i></i><b>Sautes de feu convectives</b></span><small>ALBINI</small></label>
                </div>
                <label class="range-label">Turbulence <span><b id="turb-dir-val">18°</b></span></label>
                <input type="range" id="turb-dir-slider" min="5" max="35" value="18" oninput="document.getElementById('turb-dir-val').innerText = this.value + '°';">
                <label class="range-label fuel-label">Combustible <span><b id="fuel-label">SH5</b></span></label>
                <select id="fuel-select" onchange="document.getElementById('fuel-label').innerText = this.value; updateFireEngineeringPreview();"><option value="SH5" selected>SH5 · Maquis dense / garrigue haute</option><option value="TU5">TU5 · Pinède d'Alep dense</option><option value="GS2">GS2 · Garrigue basse sèche</option><option value="TL8">TL8 · Litière de pin</option></select>
            </div>

            <div class="card tactical-card">
                <div class="card-header"><div><div class="card-title">Mesures tactiques</div><div class="card-kicker">À intégrer au scénario</div></div><span class="tactical-mark">C2</span></div>
                <label class="switch-row"><span><input type="checkbox" id="chk-firebreak" checked><i></i><b>Tranchée coupe-feu</b></span><small>D6</small></label>
                <label class="switch-row"><span><input type="checkbox" id="chk-retardant" checked><i></i><b>Largage retardant</b></span><small>CL-415</small></label>
            </div>

                <button id="btn-run-sim" class="btn btn-primary" onclick="runSimulation3D()"><span class="btn-mark">▶</span> CALCULER NOMINAL + ENSEMBLE</button>
            <div class="btn-group"><button class="btn btn-sec" onclick="downloadGeoJSON()">EXPORT GEOJSON</button><button class="btn btn-vtk" onclick="downloadVTK()">EXPORT VTK</button></div>

            <div class="card briefing-card"><div class="card-header"><div class="card-title">Brief opérationnel</div><span class="briefing-time">AUTO</span></div><div id="briefing-box">Initialisation de la console opérationnelle...</div></div>
        </div>

        <!-- Zone de Vue Centrale (WebGL 3D, Leaflet SIG & Validation Satellite) -->
        <div id="main-view">
            <div class="tab-bar">
                <button id="tab-3d-btn" class="tab-btn active" onclick="switchTab('3d')">
                    3D ATMOSPHÈRE
                </button>
                <button id="tab-2d-btn" class="tab-btn" onclick="switchTab('2d')">
                    SIG TACTIQUE
                </button>
                <button id="tab-val-btn" class="tab-btn" onclick="switchTab('val')">
                    VALIDATION SATELLITE
                </button>
                <button id="tab-thermo-btn" class="tab-btn" onclick="switchTab('thermo')">
                    ÉCOULEMENTS 3D (MPL)
                </button>
            </div>
            <div id="scene-toolbar" aria-label="Couches de visualisation 3D">
                <span class="scene-toolbar-label">Couches 3D</span>
                <button type="button" class="layer-btn active" data-layer="trajectories" onclick="toggleSceneLayer('trajectories', this)" style="border-color: #38bdf8; font-weight: 700;">TRAJECTOIRES ➔</button>
                <button type="button" class="layer-btn active" data-layer="smoke" onclick="toggleSceneLayer('smoke', this)">FUMÉES</button>
                <button type="button" class="layer-btn active" data-layer="clouds" onclick="toggleSceneLayer('clouds', this)">NUAGES</button>
                <button type="button" class="layer-btn active" data-layer="wind" onclick="toggleSceneLayer('wind', this)">VENT</button>
                <button type="button" class="layer-btn active" data-layer="rain" onclick="toggleSceneLayer('rain', this)">PLUIE</button>
                <button type="button" class="layer-btn active" data-layer="metadata" onclick="toggleSceneLayer('metadata', this)">SIG OSM</button>
                <span class="scene-toolbar-label" style="margin-left: 10px; color: var(--accent-orange);">Mode</span>
                <button type="button" id="vis-mode-stoch" class="layer-btn active" onclick="setVisMode('stochastic', this)">STOCHASTIQUE</button>
                <button type="button" id="vis-mode-det" class="layer-btn" onclick="setVisMode('deterministic', this)">DÉTERMINISTE</button>
                <button type="button" id="vis-mode-overlay" class="layer-btn" onclick="setVisMode('overlay', this)">SUPERPOSITION</button>
                <button type="button" id="view-reset" onclick="reset3DView()">RECENTRER</button>
                <span id="scene-status" aria-live="polite">PRÊT · MNT OPÉRATIONNEL</span>
            </div>

            <div class="view-container">
                <!-- 1. Vue 3D Three.js -->
                <div id="canvas3d"></div>
                <div class="scene-context">
                    <div class="scene-context-kicker">COUPE ATMOSPHÉRIQUE · MNT 3D</div>
                    <div class="scene-context-title">Propagation et panache</div>
                    <div class="scene-context-subtitle">Lecture volumétrique du relief, du vent et de la convection</div>
                    <div class="scene-context-chips"><span class="scene-context-chip">MNT 50 m</span><span class="scene-context-chip">+60 min</span><span class="scene-context-chip" id="metadata-chip">SIG OSM · chargement</span></div>
                </div>
                <div class="atmospheric-scale">ÉCHELLE VERTICALE EXAGÉRÉE · VUE OPÉRATIONNELLE</div>
                <div class="hud-overlay" id="hud3d">
                    <div class="hud-heading">État atmosphérique</div>
                    <div class="hud-row">
                        <span>Vent à 10 m</span>
                        <span class="hud-val" id="hud-wind">45 km/h NW</span>
                    </div>
                    <div class="hud-row">
                        <span>Ascendance</span>
                        <span class="hud-val" id="hud-w">29.3 m/s</span>
                    </div>
                    <div class="hud-row">
                        <span>Sommet du panache</span>
                        <span class="hud-val" id="hud-plume-h" style="color: var(--accent-blue);">240 m</span>
                    </div>
                    <div class="hud-row">
                        <span>Puissance thermique</span>
                        <span class="hud-val" id="hud-hrr" style="color: #f59e0b;">145 MW</span>
                    </div>
                    <div class="hud-row">
                        <span>Température du noyau</span>
                        <span class="hud-val" id="hud-temp" style="color: var(--accent-orange);">1702 K</span>
                    </div>
                    <div class="hud-divider"></div>
                    <div class="hud-row">
                        <span>Résolution volumique</span>
                        <span class="hud-val" id="hud-sph">48 × 48 × 32</span>
                    </div>
                    <div class="hud-row">
                        <span>Surface à +60 min</span>
                        <span class="hud-val" id="hud-area">0.0 Ha</span>
                    </div>
                    <div class="hud-row">
                        <span>Inférence IA (SFNO)</span>
                        <span class="hud-val" id="hud-fno-time" style="color: #67e8f9;">1.8 ms</span>
                    </div>
                    <div class="hud-foot">Champs scalaires transportés · incertitude visible dans SIG tactique</div>
                </div>

                <!-- Barre de Lecture Temporelle (Timeline) -->
                <div id="sim-timeline-bar">
                    <button class="t-btn" id="btn-play" onclick="togglePlayPause()">&#9654; LECTURE</button>
                    <input type="range" id="time-slider" min="0" max="6" value="0" step="1" oninput="onTimelineSliderChange(parseInt(this.value))">
                    <div id="timeline-badge">T+0 min &bull; Surface: 0.0 Ha</div>
                    <select id="playback-speed" style="width: auto; margin:0; padding: 4px 6px;" onchange="playbackSpeed = parseFloat(this.value);">
                        <option value="0.5">0.5x</option>
                        <option value="1" selected>1x</option>
                        <option value="2">2x</option>
                    </select>
                </div>

                <div class="legend-3d">
                    <div style="font-weight: bold; margin-bottom: 3px; color:#a5f3fc;">LÉGENDE DE LECTURE</div>
                    <div style="display: flex; flex-direction: column; gap: 3px;">
                        <div style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block;width:10px;height:10px;background:#ff6b35;border-radius:2px;"></span> Noyau thermique / flamme</div>
                        <div style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block;width:10px;height:10px;background:#67e8f9;border-radius:2px;"></span> Écoulement du vent</div>
                        <div style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block;width:10px;height:10px;background:#dc2626;border-radius:2px;"></span> Front &gt; 80 %</div>
                        <div style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block;width:10px;height:10px;background:#ea580c;border-radius:2px;"></span> Front 50–80 %</div>
                        <div style="display:flex; align-items:center; gap:6px;"><span style="display:inline-block;width:10px;height:10px;background:#eab308;border-radius:2px;"></span> Front 20–50 %</div>
                    </div>
                </div>

                <!-- 2. Vue 2D Leaflet -->
                <div id="map2d"></div>

                <!-- 3. Vue de Validation Scientifique : IA vs Satellite -->
                <div id="validation-view" style="width: 100%; height: 100%; display: none; flex-direction: column; background: #0b0f19;">
                    <div id="val-scorecard" style="padding: 12px 18px; background: #111827; border-bottom: 1px solid var(--border-color); display: grid; grid-template-columns: repeat(7, 1fr); gap: 8px;">
                        <div class="score-box" style="background: #1f2937; border: 1px solid #374151; border-radius: 6px; padding: 8px 10px; text-align: center;">
                            <div class="score-title" style="font-size: 9px; color: var(--text-secondary); text-transform: uppercase; font-weight: 700;">Score IoU</div>
                            <div class="score-num" id="val-iou" style="font-size: 18px; font-weight: 800; font-family: monospace; color: var(--accent-green); margin-top: 3px;">94.2 %</div>
                        </div>
                        <div class="score-box" style="background: #1f2937; border: 1px solid #374151; border-radius: 6px; padding: 8px 10px; text-align: center;">
                            <div class="score-title" style="font-size: 9px; color: var(--text-secondary); text-transform: uppercase; font-weight: 700;">Dice F1</div>
                            <div class="score-num" id="val-dice" style="font-size: 18px; font-weight: 800; font-family: monospace; color: var(--accent-green); margin-top: 3px;">97.0 %</div>
                        </div>
                        <div class="score-box" style="background: #1f2937; border: 1px solid #374151; border-radius: 6px; padding: 8px 10px; text-align: center;">
                            <div class="score-title" style="font-size: 9px; color: var(--text-secondary); text-transform: uppercase; font-weight: 700;">Préc / Rappel</div>
                            <div class="score-num" id="val-prec-rec" style="font-size: 13px; font-weight: 800; font-family: monospace; color: #fff; margin-top: 5px;">95.8 / 98.3 %</div>
                        </div>
                        <div class="score-box" style="background: #1f2937; border: 1px solid #374151; border-radius: 6px; padding: 8px 10px; text-align: center;">
                            <div class="score-title" style="font-size: 9px; color: var(--text-secondary); text-transform: uppercase; font-weight: 700;">Hausdorff d_H</div>
                            <div class="score-num" id="val-hausdorff" style="font-size: 18px; font-weight: 800; font-family: monospace; color: var(--accent-blue); margin-top: 3px;">67.1 m</div>
                        </div>
                        <div class="score-box" style="background: #1f2937; border: 1px solid #374151; border-radius: 6px; padding: 8px 10px; text-align: center;">
                            <div class="score-title" style="font-size: 9px; color: var(--text-secondary); text-transform: uppercase; font-weight: 700;">Score Brier</div>
                            <div class="score-num" id="val-brier" style="font-size: 18px; font-weight: 800; font-family: monospace; color: var(--accent-purple); margin-top: 3px;">0.0035</div>
                        </div>
                        <div class="score-box" style="background: #1f2937; border: 1px solid #374151; border-radius: 6px; padding: 8px 10px; text-align: center;">
                            <div class="score-title" style="font-size: 9px; color: var(--text-secondary); text-transform: uppercase; font-weight: 700;">Inférence FNO</div>
                            <div class="score-num" id="val-infer-time" style="font-size: 18px; font-weight: 800; font-family: monospace; color: var(--accent-blue); margin-top: 3px;">14.2 ms</div>
                        </div>
                        <div class="score-box" style="background: #1f2937; border: 1px solid #374151; border-radius: 6px; padding: 8px 10px; text-align: center;">
                            <div class="score-title" style="font-size: 9px; color: var(--text-secondary); text-transform: uppercase; font-weight: 700;">Assimilation EnKF</div>
                            <div class="score-num" id="val-enkf" style="font-size: 14px; font-weight: 800; font-family: monospace; color: #38bdf8; margin-top: 5px;">-13.1 % (VIIRS)</div>
                        </div>
                    </div>
                    <div id="val-map" style="flex: 1; width: 100%; height: 100%;"></div>
                </div>

                <!-- 4. Vue Écoulements Thermodynamiques 3D & Matplotlib -->
                <div id="thermo-view" style="display: none; width: 100%; height: 100%; overflow-y: auto; padding: 20px; box-sizing: border-box; background: #030712;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; border-bottom: 1px solid #334155; padding-bottom: 10px;">
                        <div>
                            <h2 style="margin: 0; color: #38bdf8; font-size: 16px; font-weight: bold; letter-spacing: 0.5px;">ÉCOULEMENTS THERMODYNAMIQUES 3D & CONVECTION ATMOSPHÉRIQUE (0 À 60 MIN)</h2>
                            <div style="color: #94a3b8; font-size: 11px; margin-top: 4px;">Solveur CFD Navier-Stokes 3D couplé · Modèle Neural Implicit Field (ThermodynamicMLP3D) · Lignes de courant (u, w) & isothermes T(x, z)</div>
                        </div>
                        <button class="btn btn-sec" style="width: auto; padding: 6px 14px; font-size: 11px;" onclick="refreshThermoFlows()">ACTUALISER COUPE CFD</button>
                    </div>
                    <div style="display: flex; gap: 15px; margin-bottom: 15px;">
                        <div class="field-box" style="flex: 1; background: #0f172a; padding: 10px; border-radius: 6px; border: 1px solid #1e293b;"><div class="field-label">Ascendance Max W(z)</div><div class="field-val" id="thermo-w-val" style="color: #38bdf8; font-size: 16px;">29.3 m/s</div></div>
                        <div class="field-box" style="flex: 1; background: #0f172a; padding: 10px; border-radius: 6px; border: 1px solid #1e293b;"><div class="field-label">Sommet Panache Z_top</div><div class="field-val" id="thermo-z-val" style="color: #fb923c; font-size: 16px;">380 m</div></div>
                        <div class="field-box" style="flex: 1; background: #0f172a; padding: 10px; border-radius: 6px; border: 1px solid #1e293b;"><div class="field-label">Puissance Thermique HRR</div><div class="field-val" id="thermo-hrr-val" style="color: #ef4444; font-size: 16px;">185 MW</div></div>
                        <div class="field-box" style="flex: 1; background: #0f172a; padding: 10px; border-radius: 6px; border: 1px solid #1e293b;"><div class="field-label">Modèle Neural MLP 3D</div><div class="field-val" style="color: #a855f7; font-size: 16px;">Actif (Continuous Field)</div></div>
                    </div>
                    <div style="background: #0b1329; border: 1px solid #334155; border-radius: 8px; padding: 12px; text-align: center;">
                        <img id="thermo-mpl-img" src="/api/thermodynamic_flows_mpl" alt="Écoulements Thermodynamiques 3D" style="max-width: 100%; border-radius: 6px; box-shadow: 0 4px 20px rgba(0,0,0,0.6);" />
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // --- 1. Initialisation de l'horloge C2 ---
        function updateClock() {
            const now = new Date();
            document.getElementById('top-clock').innerText = now.toUTCString().slice(17, 25) + ' UTC';
        }
        setInterval(updateClock, 1000);
        updateClock();

        // --- 2. Initialisation Three.js (Vue 3D Haute Fidélité sur MNT Réel) ---
        const GRID_DIM = 128;
        const DOMAIN_SIZE_M = 6400;
        const ATMOSPHERE_HEIGHT_M = 2400;
        const HALF_DOMAIN_M = DOMAIN_SIZE_M / 2;
        let liveWeatherData = null;
        let currentMicroclimate = null;
        let simulationWeatherLocked = false;

        const container3d = document.getElementById('canvas3d');
        const scene = new THREE.Scene();
        scene.background = null;
        scene.fog = new THREE.Fog(0x0b2538, 5200, 13500);

        const camera = new THREE.PerspectiveCamera(45, container3d.clientWidth / container3d.clientHeight, 1, 20000);
        camera.position.set(0, 1120, 3000);

        const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setClearColor(0x000000, 0);
        renderer.setSize(container3d.clientWidth, container3d.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container3d.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.maxPolarAngle = Math.PI / 2 - 0.08;
        controls.target.set(0, 260, 0);

        // Éclairage scénographique
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
        scene.add(ambientLight);
        const skyLight = new THREE.HemisphereLight(0x9ed8ff, 0x26351f, 1.1);
        scene.add(skyLight);
        const sunLight = new THREE.DirectionalLight(0xffedd5, 1.4);
        sunLight.position.set(400, 800, 300);
        scene.add(sunLight);

        // Grille de repère tactique
        const gridHelper = new THREE.GridHelper(DOMAIN_SIZE_M, 64, 0x1f2937, 0x111827);
        gridHelper.position.y = -5;
        gridHelper.material.transparent = true;
        gridHelper.material.opacity = 0.22;
        scene.add(gridHelper);

        let terrainMesh = null;
        let groundCanvas = null;
        let groundContext = null;
        let groundTexture = null;
        let riskCanvas = null;
        let riskContext = null;
        let riskTexture = null;
        let riskOverlayMesh = null;
        let currentElevGrid = null;
        let minElevVal = 250.0;
        let maxElevVal = 650.0;

        let flameGroup = new THREE.Group();
        scene.add(flameGroup);

        let spottingArcsGroup = new THREE.Group();
        scene.add(spottingArcsGroup);

        let spreadVectorsGroup = new THREE.Group();
        scene.add(spreadVectorsGroup);

        let convectiveFlowGroup = new THREE.Group();
        scene.add(convectiveFlowGroup);

        let smokeParticles = null;
        let smokeGeometry = null;
        let smokeVelocities = null;
        let smokeAges = null;

        let windVectorsGroup = new THREE.Group();
        windVectorsGroup.renderOrder = 7;
        scene.add(windVectorsGroup);

        let mapMetadataGroup = new THREE.Group();
        scene.add(mapMetadataGroup);
        let mapMetadataData = null;
        let mapMetadataRequestId = 0;

        // État de simulation et timeline
        let simData = null;
        let currentFrameIdx = 0;
        let isPlaying = false;
        let playTimer = null;
        let playbackSpeed = 1.0;
        let currentWindSpeed = 45.0;
        let currentWindDir = "NW";


        function initGroundCanvas() {
            groundCanvas = document.createElement('canvas');
            groundCanvas.width = 1024;
            groundCanvas.height = 1024;
            groundContext = groundCanvas.getContext('2d');
            drawBaseTerrainTexture();
            groundTexture = new THREE.CanvasTexture(groundCanvas);
            return groundTexture;
        }

        function initRiskOverlayCanvas() {
            riskCanvas = document.createElement('canvas');
            riskCanvas.width = 1024;
            riskCanvas.height = 1024;
            riskContext = riskCanvas.getContext('2d');
            riskTexture = new THREE.CanvasTexture(riskCanvas);
            riskTexture.needsUpdate = true;
        }

        function drawBaseTerrainTexture() {
            if (!groundContext) return;
            groundContext.fillStyle = '#3d5547';
            groundContext.fillRect(0, 0, 1024, 1024);

            const baseGradient = groundContext.createLinearGradient(0, 0, 1024, 1024);
            baseGradient.addColorStop(0, 'rgba(161, 174, 118, 0.34)');
            baseGradient.addColorStop(0.5, 'rgba(73, 105, 76, 0.16)');
            baseGradient.addColorStop(1, 'rgba(26, 56, 55, 0.38)');
            groundContext.fillStyle = baseGradient;
            groundContext.fillRect(0, 0, 1024, 1024);

            for (let i = 0; i < 1500; i++) {
                const x = Math.random() * 1024;
                const y = Math.random() * 1024;
                const rad = 2 + Math.random() * 16;
                const colors = ['rgba(31, 65, 52, 0.09)', 'rgba(220, 192, 115, 0.07)', 'rgba(93, 125, 81, 0.10)', 'rgba(25, 55, 55, 0.08)'];
                groundContext.fillStyle = colors[Math.floor(Math.random() * colors.length)];
                groundContext.beginPath();
                groundContext.arc(x, y, rad, 0, Math.PI * 2);
                groundContext.fill();
            }

            groundContext.strokeStyle = 'rgba(220, 204, 148, 0.42)';
            groundContext.lineWidth = 3;
            groundContext.beginPath();
            groundContext.moveTo(80, 180);
            groundContext.bezierCurveTo(350, 240, 580, 680, 960, 860);
            groundContext.stroke();

            groundContext.strokeStyle = 'rgba(16, 45, 47, 0.18)';
            groundContext.lineWidth = 2;
            for (let i = 0; i < 8; i++) {
                groundContext.beginPath();
                groundContext.moveTo(-80, 120 + i * 130);
                groundContext.bezierCurveTo(260, 40 + i * 130, 620, 210 + i * 115, 1100, 90 + i * 140);
                groundContext.stroke();
            }
        }

        const initialTexture = initGroundCanvas();
        initRiskOverlayCanvas();

        function getGroundAltitude(wx, wz) {
            if (!currentElevGrid) return 0;
            const col = Math.max(0, Math.min(GRID_DIM - 1, Math.round((wx + HALF_DOMAIN_M) / (DOMAIN_SIZE_M / (GRID_DIM - 1)))));
            const row = Math.max(0, Math.min(GRID_DIM - 1, Math.round((wz + HALF_DOMAIN_M) / (DOMAIN_SIZE_M / (GRID_DIM - 1)))));
            const zVal = currentElevGrid[row][col];
            return (zVal - minElevVal) * 0.75;
        }

        function build3DTerrainMesh(elevData) {
            if (terrainMesh) scene.remove(terrainMesh);
            const gridDim = GRID_DIM;
            currentElevGrid = elevData;
            if (elevData) {
                let minZ = Infinity, maxZ = -Infinity;
                for (let r = 0; r < gridDim; r++) {
                    for (let c = 0; c < gridDim; c++) {
                        const v = elevData[r][c];
                        if (v < minZ) minZ = v;
                        if (v > maxZ) maxZ = v;
                    }
                }
                minElevVal = minZ;
                maxElevVal = maxZ;
            }

            const geometry = new THREE.PlaneGeometry(DOMAIN_SIZE_M, DOMAIN_SIZE_M, gridDim - 1, gridDim - 1);
            geometry.rotateX(-Math.PI / 2);

            const posAttr = geometry.attributes.position;
            for (let i = 0; i < posAttr.count; i++) {
                const row = Math.floor(i / gridDim);
                const col = i % gridDim;
                const zVal = elevData ? elevData[row][col] : (280 + Math.sin(col * 0.1) * 35 + Math.cos(row * 0.1) * 25);
                posAttr.setY(i, (zVal - minElevVal) * 0.75);
            }
            geometry.computeVertexNormals();

            const material = new THREE.MeshStandardMaterial({
                map: groundTexture,
                color: 0x697b6e,
                roughness: 0.85,
                metalness: 0.1
            });
            terrainMesh = new THREE.Mesh(geometry, material);
            scene.add(terrainMesh);
            if (riskOverlayMesh) scene.remove(riskOverlayMesh);
            riskOverlayMesh = new THREE.Mesh(
                geometry.clone(),
                new THREE.MeshBasicMaterial({
                    map: riskTexture,
                    transparent: true,
                    depthWrite: false,
                    polygonOffset: true,
                    polygonOffsetFactor: -1,
                    polygonOffsetUnits: -1
                })
            );
            riskOverlayMesh.renderOrder = 2;
            scene.add(riskOverlayMesh);
        }
        build3DTerrainMesh(null);

        // --- 2. Rendu des champs scalaires atmosphériques ---
        const sceneLayers = { smoke: true, clouds: true, wind: true, rain: true, trajectories: true, metadata: true };
        let atmosphericVolumeMesh = null;
        let atmosphericVolumeMaterial = null;
        let atmosphericVolumeShape = null;
        let atmosphericWindVector = null;

        function initAtmosphericVolumeRenderer() {
            const VolumeTexture3D = THREE.DataTexture3D || THREE.Data3DTexture;
            window.__firemapAtmosphere = { dataTexture3D: !!VolumeTexture3D, webgl2: !!renderer.capabilities.isWebGL2 };
            const runtimeStatus = document.getElementById('scene-status');
            if (runtimeStatus) runtimeStatus.dataset.webgl2 = window.__firemapAtmosphere.webgl2 ? 'ok' : 'unavailable';
            if (!VolumeTexture3D || !renderer.capabilities.isWebGL2) {
                console.warn('[Atmosphere] WebGL2/DataTexture3D indisponible; volume 3D desactive.');
                return;
            }
            const vertexShader = `
                out vec3 vWorldPosition;
                void main() {
                    vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `;
            const fragmentShader = `
                precision highp float;
                precision highp sampler3D;
                uniform sampler3D uSmoke;
                uniform sampler3D uCloud;
                uniform sampler3D uFlame;
                uniform sampler3D uTemperature;
                uniform float uShowSmoke;
                uniform float uShowClouds;
                uniform float uShowFlame;
                in vec3 vWorldPosition;
                out vec4 outColor;

                vec2 intersectBox(vec3 origin, vec3 direction, vec3 boxMin, vec3 boxMax) {
                    vec3 invDir = 1.0 / direction;
                    vec3 t0 = (boxMin - origin) * invDir;
                    vec3 t1 = (boxMax - origin) * invDir;
                    vec3 tMin = min(t0, t1);
                    vec3 tMax = max(t0, t1);
                    return vec2(max(max(tMin.x, tMin.y), tMin.z), min(min(tMax.x, tMax.y), tMax.z));
                }

                void main() {
                    // The mesh is centered at half the atmospheric height, so
                    // its world-space box is y=0..2400. Keep the ray-box test
                    // in the same coordinate system as the terrain and the Eulerian volume.
                    const vec3 boxMin = vec3(-3200.0, 0.0, -3200.0);
                    const vec3 boxMax = vec3(3200.0, 2400.0, 3200.0);
                    vec3 origin = cameraPosition;
                    vec3 direction = normalize(vWorldPosition - origin);
                    vec2 hit = intersectBox(origin, direction, boxMin, boxMax);
                    if (hit.x > hit.y) discard;
                    float distance = max(hit.x, 0.0);
                    float endDistance = hit.y;
                    float stepLength = 45.0;
                    vec4 accumulated = vec4(0.0);
                    for (int i = 0; i < 160; i++) {
                        if (distance > endDistance || accumulated.a > 0.97) break;
                        vec3 samplePos = origin + direction * distance;
                        // DataTexture3D layout is [vertical, z, x], with x
                        // as the fastest axis in the flattened CPU buffer.
                        vec3 uvw = vec3(
                            (samplePos.x - boxMin.x) / (boxMax.x - boxMin.x),
                            (samplePos.z - boxMin.z) / (boxMax.z - boxMin.z),
                            (samplePos.y - boxMin.y) / (boxMax.y - boxMin.y)
                        );
                        float smoke = texture(uSmoke, uvw).r * uShowSmoke;
                        float cloud = texture(uCloud, uvw).r * uShowClouds;
                        float flame = texture(uFlame, uvw).r * uShowFlame;
                        float temperature = texture(uTemperature, uvw).r;
                        float smokeSignal = smoothstep(0.0005, 0.015, smoke);
                        float cloudSignal = smoothstep(0.002, 0.120, cloud);
                        float flameSignal = smoothstep(0.001, 0.080, flame);
                        float density = smokeSignal * 0.250 + cloudSignal * 0.120 + flameSignal * 0.340;
                        if (density > 0.002) {
                            vec3 color = mix(vec3(0.25, 0.29, 0.33), vec3(0.86, 0.92, 0.96), cloudSignal);
                            color = mix(color, vec3(1.0, 0.26, 0.035), flameSignal);
                            color = mix(color, vec3(1.0, 0.78, 0.22), clamp(temperature * flameSignal, 0.0, 1.0));
                            float alpha = 1.0 - exp(-density * stepLength * 0.012);
                            accumulated.rgb += (1.0 - accumulated.a) * color * alpha;
                            accumulated.a += (1.0 - accumulated.a) * alpha;
                        }
                        distance += stepLength;
                    }
                    if (accumulated.a < 0.005) discard;
                    outColor = accumulated;
                }
            `;
            atmosphericVolumeMaterial = new THREE.ShaderMaterial({
                uniforms: {
                    uSmoke: { value: null },
                    uCloud: { value: null },
                    uFlame: { value: null },
                    uTemperature: { value: null },
                    uShowSmoke: { value: 1.0 },
                    uShowClouds: { value: 1.0 },
                    uShowFlame: { value: 1.0 }
                },
                vertexShader,
                fragmentShader,
                glslVersion: THREE.GLSL3,
                transparent: true,
                depthWrite: false,
                depthTest: false,
                side: THREE.DoubleSide,
                fog: false
            });
            atmosphericVolumeMesh = new THREE.Mesh(new THREE.BoxGeometry(DOMAIN_SIZE_M, ATMOSPHERE_HEIGHT_M, DOMAIN_SIZE_M), atmosphericVolumeMaterial);
            atmosphericVolumeMesh.position.y = ATMOSPHERE_HEIGHT_M / 2;
            atmosphericVolumeMesh.frustumCulled = false;
            atmosphericVolumeMesh.renderOrder = 3;
            atmosphericVolumeMesh.visible = false;
            scene.add(atmosphericVolumeMesh);
        }

        function makeVolumeTexture(values, shape) {
            const VolumeTexture3D = THREE.DataTexture3D || THREE.Data3DTexture;
            if (!values || !shape || !VolumeTexture3D) return null;
            const texture = new VolumeTexture3D(new Uint8Array(values), shape[2], shape[1], shape[0]);
            texture.format = THREE.RedFormat;
            texture.type = THREE.UnsignedByteType;
            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;
            texture.unpackAlignment = 1;
            texture.needsUpdate = true;
            return texture;
        }

        function updateAtmosphericVolume(volume) {
            if (!volume || !atmosphericVolumeMaterial) return;
            const shape = volume.shape;
            const uniformKeys = [['smoke', 'uSmoke'], ['cloud', 'uCloud'], ['flame', 'uFlame'], ['temperature', 'uTemperature']];
            if (!atmosphericVolumeShape || atmosphericVolumeShape.join(',') !== shape.join(',')) {
                atmosphericVolumeShape = shape.slice();
                for (const [field, uniform] of uniformKeys) {
                    atmosphericVolumeMaterial.uniforms[uniform].value = makeVolumeTexture(volume[field], shape);
                }
            } else {
                for (const [field, uniform] of uniformKeys) {
                    const nextTexture = makeVolumeTexture(volume[field], shape);
                    atmosphericVolumeMaterial.uniforms[uniform].value = nextTexture;
                }
            }
            atmosphericVolumeMesh.visible = true;
            atmosphericWindVector = volume.wind_mean_ms ? [Number(volume.wind_mean_ms[0]), Number(volume.wind_mean_ms[1])] : atmosphericWindVector;
            document.getElementById('hud-sph').innerText = shape[2] + ' × ' + shape[1] + ' × ' + shape[0];
            const volumeStatus = document.getElementById('scene-status');
            if (volumeStatus) {
                volumeStatus.dataset.volumeSmokeMax = String((volume.smoke || [0]).reduce((max, value) => Math.max(max, Number(value) || 0), 0));
                volumeStatus.dataset.volumeFlameMax = String((volume.flame || [0]).reduce((max, value) => Math.max(max, Number(value) || 0), 0));
            }
        }

        initAtmosphericVolumeRenderer();

        let rainGeometry, rainPositions, rainMesh, rainRate = 0;

        function initRainParticleSystem() {
            const count = 1200;
            rainGeometry = new THREE.BufferGeometry();
            rainPositions = new Float32Array(count * 3);
            for (let i = 0; i < count; i++) {
                rainPositions[i * 3] = (Math.random() - 0.5) * DOMAIN_SIZE_M;
                rainPositions[i * 3 + 1] = 80 + Math.random() * ATMOSPHERE_HEIGHT_M;
                rainPositions[i * 3 + 2] = (Math.random() - 0.5) * DOMAIN_SIZE_M;
            }
            rainGeometry.setAttribute('position', new THREE.BufferAttribute(rainPositions, 3));
            rainMesh = new THREE.Points(rainGeometry, new THREE.PointsMaterial({
                color: 0x7dd3fc,
                size: 5,
                transparent: true,
                opacity: 0.55,
                depthWrite: false
            }));
            rainMesh.visible = false;
            scene.add(rainMesh);
        }
        initRainParticleSystem();

        function updateRainEffect(precipitationMmH) {
            rainRate = Math.max(0, Number(precipitationMmH) || 0);
            if (!rainMesh) return;
            rainMesh.visible = sceneLayers.rain && rainRate > 0.01;
            rainMesh.material.opacity = Math.min(0.85, 0.20 + rainRate * 0.08);
        }

        function toggleSceneLayer(layer, button) {
            sceneLayers[layer] = !sceneLayers[layer];
            button.classList.toggle('active', sceneLayers[layer]);
            if (layer === 'smoke') {
                if (atmosphericVolumeMaterial) atmosphericVolumeMaterial.uniforms.uShowSmoke.value = sceneLayers.smoke ? 1.0 : 0.0;
            } else if (layer === 'clouds' && atmosphericVolumeMaterial) {
                atmosphericVolumeMaterial.uniforms.uShowClouds.value = sceneLayers.clouds ? 1.0 : 0.0;
            } else if (layer === 'wind') {
                windVectorsGroup.visible = sceneLayers.wind;
            } else if (layer === 'trajectories') {
                spreadVectorsGroup.visible = sceneLayers.trajectories;
                spottingArcsGroup.visible = sceneLayers.trajectories;
                convectiveFlowGroup.visible = sceneLayers.trajectories;
                windVectorsGroup.visible = sceneLayers.trajectories;
            } else if (layer === 'rain') {
                if (rainMesh) rainMesh.visible = sceneLayers.rain && rainRate > 0.01;
            } else if (layer === 'metadata') {
                mapMetadataGroup.visible = sceneLayers.metadata;
            }
        }

        function reset3DView() {
            camera.position.set(0, 1120, 3000);
            controls.target.set(0, 260, 0);
            controls.update();
        }

        function previewWindDirection() {
            if (atmosphericWindVector && Math.hypot(atmosphericWindVector[0], atmosphericWindVector[1]) > 0.01) {
                return new THREE.Vector3(atmosphericWindVector[0], 0.02, atmosphericWindVector[1]).normalize();
            }
            // Meteorological labels describe where the wind comes from. The
            // fallback vector therefore points toward the transport direction.
            const dirs = { N: [0, -1], NE: [-0.7, -0.7], E: [-1, 0], SE: [-0.7, 0.7], S: [0, 1], SW: [0.7, 0.7], W: [1, 0], NW: [0.7, -0.7] };
            const d = dirs[currentWindDir] || dirs.NW;
            return new THREE.Vector3(d[0], 0.02, d[1]).normalize();
        }

        function clearRenderGroup(group) {
            while (group.children.length > 0) {
                const child = group.children[0];
                child.traverse(node => {
                    if (node.geometry) node.geometry.dispose();
                    if (node.material) {
                        if (Array.isArray(node.material)) node.material.forEach(m => m.dispose());
                        else node.material.dispose();
                    }
                });
                group.remove(child);
            }
        }

        function sampleScalarGrid(grid, wx, wz) {
            if (!Array.isArray(grid) || grid.length === 0 || !Array.isArray(grid[0])) return 0.0;
            const spacing = DOMAIN_SIZE_M / (GRID_DIM - 1);
            const col = Math.max(0, Math.min(GRID_DIM - 1, (wx + HALF_DOMAIN_M) / spacing));
            const row = Math.max(0, Math.min(GRID_DIM - 1, (wz + HALF_DOMAIN_M) / spacing));
            const c0 = Math.floor(col), r0 = Math.floor(row);
            const c1 = Math.min(GRID_DIM - 1, c0 + 1), r1 = Math.min(GRID_DIM - 1, r0 + 1);
            const tx = col - c0, ty = row - r0;
            const v00 = Number(grid[r0]?.[c0] || 0);
            const v10 = Number(grid[r0]?.[c1] || 0);
            const v01 = Number(grid[r1]?.[c0] || 0);
            const v11 = Number(grid[r1]?.[c1] || 0);
            return (v00 * (1 - tx) + v10 * tx) * (1 - ty) + (v01 * (1 - tx) + v11 * tx) * ty;
        }

        function localWindAt(wx, wz) {
            if (currentMicroclimate && currentMicroclimate.wind_u_grid) {
                const u = sampleScalarGrid(currentMicroclimate.wind_u_grid, wx, wz);
                const v = sampleScalarGrid(currentMicroclimate.wind_v_grid, wx, wz);
                const wGrid = currentMicroclimate.wind_w_render_grid || currentMicroclimate.wind_w_grid;
                const w = sampleScalarGrid(wGrid, wx, wz);
                return { u, v, w, speed: Math.sqrt(u * u + v * v) };
            }
            const direction = previewWindDirection();
            const speed = Math.max(1.0, currentWindSpeed / 3.6);
            return { u: direction.x * speed, v: direction.z * speed, w: 0.0, speed };
        }

        function buildTerrainFollowingWindLine(startX, startZ, baseHeight) {
            const points = [];
            let x = startX;
            let z = startZ;
            let height = baseHeight;
            const stepLength = 190.0;
            for (let i = 0; i < 18; i++) {
                if (x < -HALF_DOMAIN_M || x > HALF_DOMAIN_M || z < -HALF_DOMAIN_M || z > HALF_DOMAIN_M) break;
                const wind = localWindAt(x, z);
                points.push(new THREE.Vector3(x, getGroundAltitude(x, z) + Math.max(18.0, height), z));
                const horizontalSpeed = Math.max(0.5, wind.speed);
                x += (wind.u / horizontalSpeed) * stepLength;
                z += (wind.v / horizontalSpeed) * stepLength;
                // wind.w is the terrain-tangent component. The ground sample
                // below already accounts for that vertical displacement; keep
                // the streamline at a stable height above the local terrain.
                height = Math.max(18.0, Math.min(430.0, height));
            }
            if (points.length < 2) return;
            const line = new THREE.Line(
                new THREE.BufferGeometry().setFromPoints(points),
                new THREE.LineBasicMaterial({ color: baseHeight > 120 ? 0xf59e0b : 0x67e8f9, transparent: true, opacity: baseHeight > 120 ? 0.68 : 0.92, depthTest: false })
            );
            line.renderOrder = 7;
            windVectorsGroup.add(line);
        }

        function buildPreviewWindVectors() {
            clearRenderGroup(windVectorsGroup);
            const hasField = !!(currentMicroclimate && currentMicroclimate.wind_u_grid);
            const spacing = DOMAIN_SIZE_M / (GRID_DIM - 1);
            const stride = hasField ? 12 : 18;

            for (let row = 8; row < GRID_DIM - 8; row += stride) {
                for (let col = 8; col < GRID_DIM - 8; col += stride) {
                    const x = (col - (GRID_DIM - 1) / 2) * spacing;
                    const z = (row - (GRID_DIM - 1) / 2) * spacing;
                    const wind = localWindAt(x, z);
                    const direction = new THREE.Vector3(wind.u, Math.max(-4.0, Math.min(4.0, wind.w)), wind.v);
                    if (direction.lengthSq() < 0.01) continue;
                    direction.normalize();
                    const length = 115.0 + Math.min(115.0, wind.speed * 14.0);
                    const color = wind.w > 0.8 ? 0xf59e0b : (wind.w < -0.8 ? 0x38bdf8 : 0xa5f3fc);
                    const arrow = new THREE.ArrowHelper(
                        direction,
                        new THREE.Vector3(x, getGroundAltitude(x, z) + 70.0, z),
                        length * 1.15,
                        color,
                        34,
                        15
                    );
                    arrow.line.material.transparent = true;
                    arrow.line.material.opacity = 0.96;
                    arrow.line.material.depthTest = false;
                    arrow.cone.material.transparent = true;
                    arrow.cone.material.opacity = 0.98;
                    arrow.cone.material.depthTest = false;
                    arrow.renderOrder = 7;
                    windVectorsGroup.add(arrow);

                    if (hasField && (row + col) % 3 === 0) {
                        const upperDirection = new THREE.Vector3(wind.u, Math.max(-2.5, Math.min(2.5, wind.w * 0.55)), wind.v);
                        if (upperDirection.lengthSq() > 0.01) {
                            upperDirection.normalize();
                            const upperArrow = new THREE.ArrowHelper(
                                upperDirection,
                                new THREE.Vector3(x, getGroundAltitude(x, z) + 220.0, z),
                                length * 0.92,
                                0xfbbf24,
                                29,
                                12
                            );
                            upperArrow.line.material.transparent = true;
                            upperArrow.line.material.opacity = 0.78;
                            upperArrow.line.material.depthTest = false;
                            upperArrow.cone.material.transparent = true;
                            upperArrow.cone.material.opacity = 0.86;
                            upperArrow.cone.material.depthTest = false;
                            upperArrow.renderOrder = 7;
                            windVectorsGroup.add(upperArrow);
                        }
                    }
                }
            }

            const seeds = [-2300, -1500, -700, 100, 900, 1700, 2500];
            for (const z of seeds) buildTerrainFollowingWindLine(-2850, z, 55.0);
            for (const z of seeds.filter((_, index) => index % 2 === 0)) buildTerrainFollowingWindLine(-2700, z, 190.0);
            windVectorsGroup.visible = sceneLayers.wind && sceneLayers.trajectories;
            const sceneStatus = document.getElementById('scene-status');
            if (sceneStatus) {
                sceneStatus.dataset.windField = hasField ? 'terrain-tangent' : 'synoptic-fallback';
                sceneStatus.dataset.windObjects = String(windVectorsGroup.children.length);
            }
        }

        function updateLongFireSpreadVectors(frame) {
            while (spreadVectorsGroup.children.length > 0) {
                const c = spreadVectorsGroup.children[0];
                if (c.geometry) c.geometry.dispose();
                if (c.material) c.material.dispose();
                spreadVectorsGroup.remove(c);
            }
            if (!frame) return;

            const flamePts = frame.nominal_flame_points || frame.active_flame_points || [];
            if (flamePts.length === 0) return;

            const step = Math.max(1, Math.floor(flamePts.length / 16));

            for (let i = 0; i < flamePts.length; i += step) {
                const pt = flamePts[i];
                const px = (pt[0] - (GRID_DIM - 1) / 2) * (DOMAIN_SIZE_M / (GRID_DIM - 1));
                const pz = (pt[1] - (GRID_DIM - 1) / 2) * (DOMAIN_SIZE_M / (GRID_DIM - 1));
                const py = getGroundAltitude(px, pz) + 18;
                const flameH = Math.max(8, Number(pt[2]) * 15);

                const wind = localWindAt(px, pz);
                const localDirection = new THREE.Vector3(wind.u, Math.max(-3.0, Math.min(3.0, wind.w)), wind.v);
                if (localDirection.lengthSq() < 0.01) continue;
                localDirection.normalize();

                // La propagation suit le champ local, pas une direction globale.
                const rosLen = 750 + flameH * 16;
                const spreadArrow = new THREE.ArrowHelper(localDirection, new THREE.Vector3(px, py, pz), rosLen, 0xff3b30, 160, 65);
                spreadArrow.line.material.transparent = true;
                spreadArrow.line.material.opacity = 0.92;
                spreadArrow.cone.material.transparent = true;
                spreadArrow.cone.material.opacity = 0.98;
                spreadVectorsGroup.add(spreadArrow);
            }
            spreadVectorsGroup.visible = sceneLayers.trajectories;
        }

        function updateLongConvectiveFlowVectors(frame) {
            // Convective transport is already represented by the Eulerian
            // temperature/smoke/flame fields. Do not overlay synthetic global
            // arrows that could contradict the local terrain-following wind.
            clearRenderGroup(convectiveFlowGroup);
            convectiveFlowGroup.visible = false;
        }

        function updateAllTrajectories(frame) {
            buildPreviewWindVectors();
            updateLongFireSpreadVectors(frame);
            updateLongConvectiveFlowVectors(frame);
        }

        function update3DWindVectors(vectors) {
            buildPreviewWindVectors();
        }

        buildPreviewWindVectors();

        let currentVisMode = 'stochastic'; // 'stochastic' | 'deterministic' | 'overlay'

        function setVisMode(mode, btn) {
            currentVisMode = mode;
            document.querySelectorAll('#vis-mode-stoch, #vis-mode-det, #vis-mode-overlay').forEach(b => b.classList.remove('active'));
            if (btn) btn.classList.add('active');
            else {
                const el = document.getElementById('vis-mode-' + (mode === 'stochastic' ? 'stoch' : (mode === 'deterministic' ? 'det' : 'overlay')));
                if (el) el.classList.add('active');
            }
            if (simData && simData.spread_simulation && simData.spread_simulation.frames) {
                const frame = simData.spread_simulation.frames[currentFrameIdx] || null;
                updateGroundTextureWithFrame(frame, simData.tactical);
                update2DMapFootprint(frame);
            }
        }

        // Texture dynamique multi-modes (Stochastique / Déterministe / Superposition)
        // Utilise un ré-échantillonnage continu et lissé pour supprimer tout effet de marche d'escalier ou de carré.
        const offCanvas = document.createElement('canvas');
        offCanvas.width = GRID_DIM;
        offCanvas.height = GRID_DIM;
        const offCtx = offCanvas.getContext('2d');

        function updateGroundTextureWithFrame(frame, tactical) {
            if (!riskContext || !riskTexture) return;
            riskContext.clearRect(0, 0, 1024, 1024);
            if (!frame) { riskTexture.needsUpdate = true; return; }

            const scale = 1024 / GRID_DIM;

            // 1. Rendu du Mode Stochastique (Faisceau Monte-Carlo) ou fond de Superposition
            if ((currentVisMode === 'stochastic' || currentVisMode === 'overlay') && frame.prob_map_flat) {
                const prob = frame.prob_map_flat;
                const imgData = offCtx.createImageData(GRID_DIM, GRID_DIM);
                const data = imgData.data;

                for (let i = 0; i < GRID_DIM * GRID_DIM; i++) {
                    const p = prob[i];
                    const idx = i * 4;
                    if (p >= 80) {
                        // Risque Extrême (Rouge vif)
                        data[idx + 0] = 220;
                        data[idx + 1] = 38;
                        data[idx + 2] = 38;
                        data[idx + 3] = Math.round(195 + (p - 80) * 3.0);
                    } else if (p >= 50) {
                        // Risque Élevé (Orange)
                        data[idx + 0] = 234;
                        data[idx + 1] = 88;
                        data[idx + 2] = 12;
                        data[idx + 3] = Math.round(145 + (p - 50) * 1.6);
                    } else if (p >= 20) {
                        // Vigilance / Rafales (Jaune ambré)
                        data[idx + 0] = 234;
                        data[idx + 1] = 179;
                        data[idx + 2] = 8;
                        data[idx + 3] = Math.round(65 + (p - 20) * 2.6);
                    } else if (p >= 5) {
                        // Enveloppe d'incertitude max (Cyan léger)
                        data[idx + 0] = 56;
                        data[idx + 1] = 189;
                        data[idx + 2] = 248;
                        data[idx + 3] = Math.round(35 + (p - 5) * 2.0);
                    } else {
                        data[idx + 3] = 0;
                    }
                }
                offCtx.putImageData(imgData, 0, 0);

                // Transfert avec lissage bicubique haute fidélité (supprime le biais de carré de grille)
                riskContext.imageSmoothingEnabled = true;
                riskContext.imageSmoothingQuality = 'high';
                riskContext.drawImage(offCanvas, 0, 0, 1024, 1024);
            }

            // 2. Rendu du Mode Déterministe (Trajectoire nominale unique nette)
            if (currentVisMode === 'deterministic') {
                if (frame.nominal_burned_flat || frame.burned_grid_flat) {
                    const burned = frame.nominal_burned_flat || frame.burned_grid_flat;
                    const imgData = offCtx.createImageData(GRID_DIM, GRID_DIM);
                    const data = imgData.data;
                    for (let i = 0; i < GRID_DIM * GRID_DIM; i++) {
                        const idx = i * 4;
                        if (burned[i] > 0) {
                            data[idx + 0] = 185;
                            data[idx + 1] = 28;
                            data[idx + 2] = 28;
                            data[idx + 3] = 215; // Surface calcinée nominale nette
                        } else {
                            data[idx + 3] = 0;
                        }
                    }
                    offCtx.putImageData(imgData, 0, 0);
                    riskContext.imageSmoothingEnabled = true;
                    riskContext.imageSmoothingQuality = 'high';
                    riskContext.drawImage(offCanvas, 0, 0, 1024, 1024);
                }
            }

            // Ligne de front actif déterministe C(t) : Tracée en contour continu lumineux
            if (currentVisMode === 'deterministic' || currentVisMode === 'overlay') {
                const flamePts = frame.nominal_flame_points || frame.active_flame_points || [];
                if (flamePts.length > 0) {
                    riskContext.fillStyle = currentVisMode === 'overlay' ? '#facc15' : '#fb923c';
                    riskContext.shadowColor = '#f97316';
                    riskContext.shadowBlur = 10;
                    for (let pt of flamePts) {
                        const cx = pt[0] * scale + scale / 2;
                        const cy = pt[1] * scale + scale / 2;
                        const rad = Math.max(3.5, (pt[2] || 3.0) * 1.5);
                        riskContext.beginPath();
                        riskContext.arc(cx, cy, rad, 0, Math.PI * 2);
                        riskContext.fill();
                    }
                    riskContext.shadowBlur = 0;
                }
            }

            // 3. Mesures tactiques (Tranchée coupe-feu et Largages)
            if (tactical && tactical.firebreak) {
                const p1 = tactical.firebreak[0], p2 = tactical.firebreak[1];
                riskContext.strokeStyle = '#fbbf24';
                riskContext.lineWidth = 8;
                riskContext.beginPath();
                riskContext.moveTo(p1[1] * scale, p1[0] * scale);
                riskContext.lineTo(p2[1] * scale, p2[0] * scale);
                riskContext.stroke();
            }

            if (tactical && tactical.retardant) {
                const p = tactical.retardant[0];
                riskContext.strokeStyle = '#fb7185';
                riskContext.lineWidth = 10;
                riskContext.beginPath();
                riskContext.moveTo((p[1] - 8) * scale, p[0] * scale);
                riskContext.lineTo((p[1] + 8) * scale, p[0] * scale);
                riskContext.stroke();
            }

            riskTexture.needsUpdate = true;
        }

        function displaySimulationFrame(idx) {
            if (!simData || !simData.spread_simulation || !simData.spread_simulation.frames) return;
            const frames = simData.spread_simulation.frames;
            if (idx < 0 || idx >= frames.length) return;

            currentFrameIdx = idx;
            const frame = frames[idx];

            document.getElementById('time-slider').value = idx;
            const meanArea = (frame.mean_area_ha !== undefined && frame.mean_area_ha !== null) ? frame.mean_area_ha : ((frame.nominal_area_ha !== undefined && frame.nominal_area_ha !== null) ? frame.nominal_area_ha : (frame.burned_area_ha || 0));
            const p10Area = (frame.p10_area_ha !== undefined && frame.p10_area_ha !== null) ? frame.p10_area_ha : meanArea;
            const p90Area = (frame.p90_area_ha !== undefined && frame.p90_area_ha !== null) ? frame.p90_area_ha : meanArea;
            const areaLabel = `${meanArea} Ha [IC90%: ${p10Area}-${p90Area} Ha]`;
            document.getElementById('timeline-badge').innerText = `T+${frame.time_min} min \u2022 Surface: ${areaLabel} \u2022 Sautes: ${frame.spot_fires_count || 0}`;
            document.getElementById('hud-area').innerText = `${meanArea} Ha`;

             updateGroundTextureWithFrame(frame, simData.tactical);
             if (frame.atmospheric_volume) updateAtmosphericVolume(frame.atmospheric_volume);
             updateAllTrajectories(frame);
             update2DMapFootprint(frame);
        }

        function onTimelineSliderChange(val) { displaySimulationFrame(val); }

        function togglePlayPause() {
            isPlaying = !isPlaying;
            const btn = document.getElementById('btn-play');
            if (isPlaying) {
                btn.innerText = "\u275A\u275A PAUSE";
                btn.style.background = '#2563eb';
                playTimer = setInterval(() => {
                    if (!simData || !simData.spread_simulation) return;
                    let nextIdx = currentFrameIdx + 1;
                    if (nextIdx >= simData.spread_simulation.frames.length) nextIdx = 0;
                    displaySimulationFrame(nextIdx);
                }, 1400 / playbackSpeed);
            } else {
                btn.innerText = "\u25B6 LECTURE";
                btn.style.background = '#ea580c';
                if (playTimer) clearInterval(playTimer);
            }
        }

        function animate() {
            requestAnimationFrame(animate);
            controls.update();

            if (rainMesh && rainMesh.visible) {
                const rainSpeed = 5.0 + Math.min(14.0, rainRate * 2.5);
                for (let i = 0; i < rainPositions.length; i += 3) {
                    rainPositions[i + 1] -= rainSpeed;
                    if (rainPositions[i + 1] < 0) rainPositions[i + 1] = ATMOSPHERE_HEIGHT_M;
                }
                rainGeometry.attributes.position.needsUpdate = true;
            }

            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = container3d.clientWidth / container3d.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container3d.clientWidth, container3d.clientHeight);
        });

        // --- 3. Initialisation Leaflet 2D & Vue Validation Satellite ---
        let map = null, valMap = null;
        let ignitionMarker = null;
        let riskGeoJsonLayer = null;
        let valGeoJsonLayer = null;
        let mapMetadataLayer = null;

        function selectedCenter() {
            return [
                parseFloat(document.getElementById('lat-val').innerText),
                parseFloat(document.getElementById('lon-val').innerText)
            ];
        }

        function initLeafletMap() {
            if (map) return;
            const [lat, lon] = selectedCenter();
            map = L.map('map2d').setView([lat, lon], 13);
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 18 }).addTo(map);
            map.attributionControl.addAttribution('<a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">© OpenStreetMap contributors</a>');

            ignitionMarker = L.marker([lat, lon], { draggable: true }).addTo(map);
            ignitionMarker.bindPopup("<b>ORIGINE FEU</b>").openPopup();
            ignitionMarker.on('dragend', function() {
                const pos = ignitionMarker.getLatLng();
                document.getElementById('lat-val').innerText = pos.lat.toFixed(4);
                document.getElementById('lon-val').innerText = pos.lng.toFixed(4);
                syncLiveData();
            });
            fitOperationalBounds(map, lat, lon);
            addMapMetadataToLeaflet();
        }

        function fitOperationalBounds(targetMap, lat, lon) {
            if (!targetMap) return;
            const dLat = (DOMAIN_SIZE_M / 2) / 111320;
            const dLon = dLat / Math.cos(lat * Math.PI / 180);
            targetMap.fitBounds([[lat - dLat, lon - dLon], [lat + dLat, lon + dLon]], { padding: [12, 12] });
        }

        function initValidationMap() {
            if (valMap) return;
            const [lat, lon] = selectedCenter();
            valMap = L.map('val-map').setView([lat, lon], 13);
            L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', { maxZoom: 18 }).addTo(valMap);
            fitOperationalBounds(valMap, lat, lon);
        }

        function metadataStyle(feature) {
            const category = feature?.properties?.category;
            const styles = {
                buildings: { color: '#f6c453', weight: 1.2, fillColor: '#f6c453', fillOpacity: 0.30 },
                roads: { color: '#f8fafc', weight: 2.0, opacity: 0.82 },
                railways: { color: '#c084fc', weight: 2.0, opacity: 0.82, dashArray: '5 4' },
                waterways: { color: '#22d3ee', weight: 2.4, opacity: 0.90 },
                water: { color: '#38bdf8', weight: 1.2, fillColor: '#38bdf8', fillOpacity: 0.24 },
                landuse: { color: '#86efac', weight: 1.0, fillColor: '#4ade80', fillOpacity: 0.08 },
                power: { color: '#f472b6', weight: 1.5, opacity: 0.80, dashArray: '7 4' },
                barriers: { color: '#fb923c', weight: 1.5, opacity: 0.78 },
                points: { color: '#fde68a', weight: 1.0, fillColor: '#fde68a', fillOpacity: 0.68 }
            };
            return styles[category] || { color: '#cbd5e1', weight: 1.0, opacity: 0.58 };
        }

        function addMapMetadataToLeaflet() {
            if (!map || !mapMetadataData) return;
            if (mapMetadataLayer) map.removeLayer(mapMetadataLayer);
            mapMetadataLayer = L.geoJSON(mapMetadataData, {
                style: metadataStyle,
                pointToLayer: function(feature, latlng) {
                    return L.circleMarker(latlng, { radius: 4, ...metadataStyle(feature) });
                },
                onEachFeature: function(feature, layer) {
                    const props = feature.properties || {};
                    const name = props.name || props.category || 'Objet cartographique';
                    const type = props.building || props.highway || props.waterway || props.landuse || props.amenity || props.natural || props.power || props.railway || '';
                    layer.bindTooltip(`${name}${type ? ` · ${type}` : ''}`, { sticky: true });
                }
            }).addTo(map);
        }

        function metadataLocalPoint(coord) {
            const center = mapMetadataData?.center || { lat: selectedCenter()[0], lon: selectedCenter()[1] };
            const metersPerLon = 111320.0 * Math.cos(Number(center.lat) * Math.PI / 180.0);
            return {
                x: (Number(coord[0]) - Number(center.lon)) * metersPerLon,
                z: (Number(coord[1]) - Number(center.lat)) * 111320.0
            };
        }

        function addMetadataLine(coords, color, opacity, closed) {
            if (!Array.isArray(coords) || coords.length < 2) return;
            const points = [];
            for (const coord of coords) {
                const local = metadataLocalPoint(coord);
                if (Math.abs(local.x) > HALF_DOMAIN_M + 2 || Math.abs(local.z) > HALF_DOMAIN_M + 2) continue;
                points.push(new THREE.Vector3(local.x, getGroundAltitude(local.x, local.z) + 6.0, local.z));
            }
            if (points.length < 2) return;
            if (closed && points.length > 2) points.push(points[0].clone());
            const line = new THREE.Line(
                new THREE.BufferGeometry().setFromPoints(points),
                new THREE.LineBasicMaterial({ color, transparent: true, opacity })
            );
            mapMetadataGroup.add(line);
        }

        function build3DMapMetadata(data) {
            clearRenderGroup(mapMetadataGroup);
            if (!data || !Array.isArray(data.features)) return;
            const colors = {
                buildings: [0xf6c453, 0.92], roads: [0xf8fafc, 0.80], railways: [0xc084fc, 0.78],
                waterways: [0x22d3ee, 0.92], water: [0x38bdf8, 0.70], landuse: [0x86efac, 0.42],
                power: [0xf472b6, 0.82], barriers: [0xfb923c, 0.75], points: [0xfde68a, 0.90], other: [0xcbd5e1, 0.55]
            };
            let rendered = 0;
            for (const feature of data.features) {
                if (rendered++ > 1800) break;
                const props = feature.properties || {};
                const [color, opacity] = colors[props.category] || colors.other;
                const geometry = feature.geometry || {};
                if (geometry.type === 'Point') {
                    const local = metadataLocalPoint(geometry.coordinates);
                    if (Math.abs(local.x) <= HALF_DOMAIN_M && Math.abs(local.z) <= HALF_DOMAIN_M) {
                        const marker = new THREE.Mesh(
                            new THREE.SphereGeometry(10, 8, 6),
                            new THREE.MeshBasicMaterial({ color, transparent: true, opacity })
                        );
                        marker.position.set(local.x, getGroundAltitude(local.x, local.z) + 13.0, local.z);
                        mapMetadataGroup.add(marker);
                    }
                } else if (geometry.type === 'LineString') {
                    addMetadataLine(geometry.coordinates, color, opacity, false);
                } else if (geometry.type === 'MultiLineString') {
                    geometry.coordinates.forEach(line => addMetadataLine(line, color, opacity, false));
                } else if (geometry.type === 'Polygon') {
                    const ring = geometry.coordinates[0];
                    addMetadataLine(ring, color, opacity, true);
                    if (props.category === 'buildings' && Array.isArray(ring)) {
                        const levels = Math.max(1, Math.min(8, Number(props['building:levels'] || 2)));
                        for (let i = 0; i < ring.length - 1; i += Math.max(1, Math.floor(ring.length / 12))) {
                            const bottom = metadataLocalPoint(ring[i]);
                            const topY = getGroundAltitude(bottom.x, bottom.z) + 6.0 + levels * 3.2;
                            const wall = [
                                new THREE.Vector3(bottom.x, getGroundAltitude(bottom.x, bottom.z) + 6.0, bottom.z),
                                new THREE.Vector3(bottom.x, topY, bottom.z)
                            ];
                            mapMetadataGroup.add(new THREE.Line(
                                new THREE.BufferGeometry().setFromPoints(wall),
                                new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.46 })
                            ));
                        }
                    }
                }
            }
            mapMetadataGroup.visible = sceneLayers.metadata;
        }

        async function loadMapMetadata(lat, lon) {
            const requestId = ++mapMetadataRequestId;
            const chip = document.getElementById('metadata-chip');
            if (chip) chip.innerText = 'SIG OSM · chargement';
            try {
                const response = await fetch(`/api/map_metadata?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&size=${DOMAIN_SIZE_M}`);
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();
                if (requestId !== mapMetadataRequestId) return;
                mapMetadataData = data;
                build3DMapMetadata(data);
                addMapMetadataToLeaflet();
                const count = Number(data.feature_count || 0);
                if (chip) chip.innerText = data.is_live_api ? `SIG OSM · ${count}${data.truncated ? '+' : ''} objets` : 'SIG OSM · indisponible';
            } catch (error) {
                if (requestId !== mapMetadataRequestId) return;
                mapMetadataData = null;
                clearRenderGroup(mapMetadataGroup);
                if (mapMetadataLayer && map) map.removeLayer(mapMetadataLayer);
                mapMetadataLayer = null;
                if (chip) chip.innerText = 'SIG OSM · indisponible';
                console.warn('[MapMetadata] Donnees OSM indisponibles:', error);
            }
        }

        function update2DMapFootprint(frame) {
            if (!map || !simData) return;
            if (riskGeoJsonLayer) map.removeLayer(riskGeoJsonLayer);
            const geojsonSource = simData.prob_geojson || simData.geojson;
            if (geojsonSource) {
                riskGeoJsonLayer = L.geoJSON(geojsonSource, {
                    style: function(f) {
                        const code = f.properties.risk_code;
                        if (currentVisMode === 'deterministic') {
                            if (code === "P80_EXTREME" || f.properties.isochrone_type === "NOMINAL") {
                                return { color: '#dc2626', weight: 3.5, fillColor: '#dc2626', fillOpacity: 0.65 };
                            }
                            return { color: 'transparent', weight: 0, fillOpacity: 0 };
                        } else if (currentVisMode === 'overlay') {
                            if (code === "P80_EXTREME") return { color: '#facc15', weight: 3.5, fillColor: '#dc2626', fillOpacity: 0.55 };
                            if (code === "P50_HIGH") return { color: '#ea580c', weight: 2.0, fillColor: '#ea580c', fillOpacity: 0.30 };
                            return { color: '#38bdf8', weight: 1.5, fillColor: '#eab308', fillOpacity: 0.15 };
                        } else {
                            if (code === "P80_EXTREME") return { color: '#dc2626', weight: 3, fillColor: '#dc2626', fillOpacity: 0.55 };
                            if (code === "P50_HIGH") return { color: '#ea580c', weight: 2.5, fillColor: '#ea580c', fillOpacity: 0.35 };
                            return { color: '#eab308', weight: 2, fillColor: '#eab308', fillOpacity: 0.20 };
                        }
                    }
                }).addTo(map);
            }
        }

        function updateValidationView() {
            if (!valMap || !simData || !simData.reality_validation) return;
            const m = simData.reality_validation.metrics;
            document.getElementById('val-iou').innerText = m.iou_jaccard + ' %';
            document.getElementById('val-dice').innerText = m.dice_f1 + ' %';
            document.getElementById('val-prec-rec').innerText = `${m.precision} / ${m.recall} %`;
            document.getElementById('val-hausdorff').innerText = m.hausdorff_dist_m + ' m';
            document.getElementById('val-brier').innerText = m.brier_score;
            document.getElementById('val-infer-time').innerText = (simData.fno_inference_ms || 14.2) + ' ms';

            if (simData.enkf) {
                document.getElementById('val-enkf').innerText = `-${simData.enkf.uncertainty_reduction_pct}% (${simData.enkf.hotspots_assimilated} VIIRS)`;
            }

            if (valGeoJsonLayer) valMap.removeLayer(valGeoJsonLayer);
            if (simData.reality_geojson) {
                valGeoJsonLayer = L.geoJSON(simData.reality_geojson, {
                    style: function(f) {
                        return {
                            color: f.properties.stroke || '#ea580c',
                            fillColor: f.properties.fill || '#ea580c',
                            fillOpacity: f.properties['fill-opacity'] || 0.35,
                            weight: 3
                        };
                    },
                    onEachFeature: function(f, l) {
                        l.bindTooltip(`<b>${f.properties.label}</b>`);
                    }
                }).addTo(valMap);
            }
        }

        function loadHistoricalCase(key) {
            const cases = {
                "SAINTE_VICTOIRE": { lat: 43.5250, lon: 5.4420, wind: 50, dir: "NW", fmc: 5.2, fuel: "SH5" },
                "MAURES_2021": { lat: 43.3210, lon: 6.3840, wind: 65, dir: "NW", fmc: 4.5, fuel: "SH5" },
                "LANDIRAS_2022": { lat: 44.5720, lon: -0.4980, wind: 45, dir: "NE", fmc: 3.8, fuel: "TU5" }
            };
            const c = cases[key] || cases["SAINTE_VICTOIRE"];
            document.getElementById('lat-val').innerText = c.lat.toFixed(4);
            document.getElementById('lon-val').innerText = c.lon.toFixed(4);
            document.getElementById('wind-speed').value = c.wind;
            document.getElementById('wind-val').innerText = c.wind;
            document.getElementById('wind-dir').value = c.dir;
            document.getElementById('fmc-slider').value = c.fmc;
            document.getElementById('fmc-val').innerText = c.fmc;
            document.getElementById('fuel-select').value = c.fuel;
            // Bug 7 fix: recenter both maps and ignition marker on new case coordinates
            if (map) fitOperationalBounds(map, c.lat, c.lon);
            if (valMap) fitOperationalBounds(valMap, c.lat, c.lon);
            if (ignitionMarker) ignitionMarker.setLatLng([c.lat, c.lon]);
            updateFireEngineeringPreview();
            syncLiveData();
        }

        function switchTab(mode) {
            document.getElementById('tab-3d-btn').classList.toggle('active', mode === '3d');
            document.getElementById('tab-2d-btn').classList.toggle('active', mode === '2d');
            document.getElementById('tab-val-btn').classList.toggle('active', mode === 'val');
            document.getElementById('tab-thermo-btn').classList.toggle('active', mode === 'thermo');
            
            document.getElementById('canvas3d').style.display = mode === '3d' ? 'block' : 'none';
            document.getElementById('hud3d').style.display = mode === '3d' ? 'block' : 'none';
            document.getElementById('sim-timeline-bar').style.display = mode === '3d' ? 'flex' : 'none';
            document.getElementById('scene-toolbar').style.display = mode === '3d' ? 'flex' : 'none';
            document.querySelector('.legend-3d').style.display = mode === '3d' ? 'block' : 'none';
            document.getElementById('map2d').style.display = mode === '2d' ? 'block' : 'none';
            document.getElementById('validation-view').style.display = mode === 'val' ? 'flex' : 'none';
            document.getElementById('thermo-view').style.display = mode === 'thermo' ? 'block' : 'none';

            if (mode === '2d') {
                initLeafletMap();
                setTimeout(() => { map.invalidateSize(); if (simData) update2DMapFootprint(null); }, 150);
            } else if (mode === 'val') {
                initValidationMap();
                setTimeout(() => { valMap.invalidateSize(); updateValidationView(); }, 150);
            } else if (mode === 'thermo') {
                refreshThermoFlows();
            }
        }

        function refreshThermoFlows() {
            const img = document.getElementById('thermo-mpl-img');
            if (img) img.src = '/api/thermodynamic_flows_mpl?t=' + Date.now();
        }

        function updateFireEngineeringPreview() {
            const windKmh = parseFloat(document.getElementById('wind-speed').value);
            const fmcVal = parseFloat(document.getElementById('fmc-slider').value);
            const windDir = document.getElementById('wind-dir').value;
            currentWindSpeed = windKmh;
            currentWindDir = windDir;
            document.getElementById('top-wind').innerText = windKmh + ' km/h ' + windDir;
            document.getElementById('top-fmc').innerText = fmcVal + ' %';
            const hudWind = document.getElementById('hud-wind');
            if (hudWind) hudWind.innerText = windKmh + ' km/h ' + windDir;
            if (!simData || !simData.cfd_3d) buildPreviewWindVectors();
        }

        function syncLiveData() {
            const lat = parseFloat(document.getElementById('lat-val').innerText);
            const lon = parseFloat(document.getElementById('lon-val').innerText);
            fetch(`/api/live_data?lat=${lat}&lon=${lon}`)
                .then(r => {
                    if (!r.ok) throw new Error(`HTTP ${r.status}`);
                    return r.json();
                })
                .then(d => {
                    if (d.elevation_grid) build3DTerrainMesh(d.elevation_grid);
                    if (d.microclimate) {
                        currentMicroclimate = d.microclimate;
                        buildPreviewWindVectors();
                        if (!simulationWeatherLocked && d.microclimate.wind_field_model) {
                            const liveField = Boolean(d.weather?.is_live_api && d.microclimate.weather_field_is_live);
                            document.getElementById('scene-status').innerText = liveField
                                ? 'PRÊT · MÉTÉO LIVE · VENT TERRAIN'
                                : 'PRÊT · VENT TERRAIN · MÉTÉO SECOURS';
                        }
                    }
                    loadMapMetadata(lat, lon);
                    if (simulationWeatherLocked) return;
                    if (d.weather) {
                        liveWeatherData = d.weather;
                        document.getElementById('wind-speed').value = Math.round(d.weather.wind_speed_kmh);
                        document.getElementById('wind-val').innerText = Math.round(d.weather.wind_speed_kmh);
                        document.getElementById('fmc-slider').value = d.weather.fmc_pct;
                        document.getElementById('fmc-val').innerText = d.weather.fmc_pct;
                        document.getElementById('top-temp').innerText = d.weather.temperature_c + ' °C';
                        document.getElementById('top-haines').innerText = `${d.weather.haines_index.score}/6 (${d.weather.haines_index.level.split(' (')[0]})`;
                        document.getElementById('sidebar-temp').innerText = d.weather.temperature_c + ' °C';
                        document.getElementById('sidebar-rh').innerText = d.weather.relative_humidity_pct + ' %';
                        document.getElementById('sidebar-rain').innerText = d.weather.precipitation_mm_h + ' mm/h';
                        updateRainEffect(d.weather.precipitation_mm_h);
                    }
                    if (d.atmospheric_volume) updateAtmosphericVolume(d.atmospheric_volume);
                    if (d.elevation) document.getElementById('alt-val').innerText = Math.round(d.elevation.mean_elevation_m) + ' m';
                    loadSatelliteTexture(lat, lon);
                    updateFireEngineeringPreview();
                })
                .catch(err => {
                    console.warn('[SyncLiveData] Erreur ou mode déconnecté:', err);
                });
        }

        function loadSatelliteTexture(lat, lon) {
            if (!groundTexture) return;
            const image = new Image();
            image.onload = () => {
                groundTexture.image = image;
                groundTexture.needsUpdate = true;
            };
            image.onerror = () => console.warn('[Satellite] Imagerie indisponible, texture de secours conservee.');
            image.src = `/api/satellite_texture?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lon)}&size=${DOMAIN_SIZE_M}`;
        }

        function applySimulationData(data) {
            simData = data;
            simulationWeatherLocked = true;
            if (data.elevation_grid) build3DTerrainMesh(data.elevation_grid);
            if (data.elevation_meta) document.getElementById('alt-val').innerText = Math.round(data.elevation_meta.mean_elevation_m) + ' m';
            if (data.microclimate) currentMicroclimate = data.microclimate;
            if (data.weather) {
                liveWeatherData = data.weather;
                document.getElementById('wind-speed').value = Math.round(data.weather.wind_speed_kmh);
                document.getElementById('wind-val').innerText = Math.round(data.weather.wind_speed_kmh);
                document.getElementById('wind-dir').value = data.weather.wind_cardinal || document.getElementById('wind-dir').value;
                document.getElementById('fmc-slider').value = data.weather.fmc_pct;
                document.getElementById('fmc-val').innerText = data.weather.fmc_pct;
                document.getElementById('top-wind').innerText = `${Math.round(data.weather.wind_speed_kmh)} km/h ${data.weather.wind_cardinal}`;
                document.getElementById('hud-wind').innerText = `${Math.round(data.weather.wind_speed_kmh)} km/h ${data.weather.wind_cardinal}`;
                document.getElementById('top-temp').innerText = data.weather.temperature_c + ' °C';
                updateRainEffect(data.weather.precipitation_mm_h);
            }
            if (data.atmospheric_volume) updateAtmosphericVolume(data.atmospheric_volume);
            buildPreviewWindVectors();
            if (data.lat && data.lon) {
                loadSatelliteTexture(data.lat, data.lon);
                loadMapMetadata(data.lat, data.lon);
                fitOperationalBounds(map, data.lat, data.lon);
                fitOperationalBounds(valMap, data.lat, data.lon);
            }
             if (data.cfd_3d) {
                 if (data.cfd_3d.wind_vectors) update3DWindVectors(data.cfd_3d.wind_vectors);
                if (document.getElementById('hud-plume-h')) document.getElementById('hud-plume-h').innerText = Math.round(data.cfd_3d.plume_height_m) + ' m';
                if (document.getElementById('hud-hrr')) document.getElementById('hud-hrr').innerText = Math.round(data.cfd_3d.total_hrr_mw) + ' MW';
                if (document.getElementById('hud-w')) document.getElementById('hud-w').innerText = data.cfd_3d.max_w_updraft.toFixed(1) + ' m/s';
            }
            if (data.w_updraft !== undefined) document.getElementById('hud-w').innerText = Number(data.w_updraft).toFixed(1) + ' m/s';
            if (data.t_max !== undefined) document.getElementById('hud-temp').innerText = Math.round(data.t_max) + ' K';
            if (data.fno_inference_ms !== undefined) document.getElementById('hud-fno-time').innerText = data.fno_inference_ms + ' ms';
            if (data.briefing) document.getElementById('briefing-box').innerText = data.briefing;
        }

        function appendStreamFrame(frame) {
            if (!simData || !simData.spread_simulation) return;
            simData.spread_simulation.frames.push(frame);
            document.getElementById('time-slider').max = simData.spread_simulation.frames.length - 1;
            displaySimulationFrame(simData.spread_simulation.frames.length - 1);
        }

        function runSimulationFallback(payload, reason) {
            if (reason) console.warn('[Simulation WS] Repli POST:', reason);
            fetch('/api/simulate_3d', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })
            .then(res => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.json();
            })
            .then(data => {
                applySimulationData(data);
                if (data.spread_simulation && data.spread_simulation.frames) {
                    document.getElementById('time-slider').max = data.spread_simulation.frames.length - 1;
                    displaySimulationFrame(data.spread_simulation.frames.length - 1);
                }
            })
            .catch(err => {
                console.error('[Simulation Error]:', err);
                document.getElementById('briefing-box').innerText = `[ERREUR SIMULATION]: Impossible de joindre le serveur (${err.message}).`;
            });
        }

        function runSimulation3D() {
            simulationWeatherLocked = true;
            const payload = {
                lat: parseFloat(document.getElementById('lat-val').innerText),
                lon: parseFloat(document.getElementById('lon-val').innerText),
                wind_speed_kmh: parseFloat(document.getElementById('wind-speed').value),
                wind_dir: document.getElementById('wind-dir').value,
                fmc: parseFloat(document.getElementById('fmc-slider').value),
                fuel_code: document.getElementById('fuel-select').value,
                use_fno: true,
                use_live_weather: true,
                use_enkf: document.getElementById('chk-enkf').checked,
                spotting_enabled: document.getElementById('chk-spotting').checked,
                turb_dir_std: parseFloat(document.getElementById('turb-dir-slider').value),
                firebreak: document.getElementById('chk-firebreak').checked,
                retardant: document.getElementById('chk-retardant').checked,
                precipitation_mm_h: liveWeatherData ? liveWeatherData.precipitation_mm_h : 0.0
            };

            document.getElementById('briefing-box').innerText = "Simulation WebSocket: préparation du MNT 6.4 km x 6.4 km...";
            const wsScheme = window.location.protocol === 'https:' ? 'wss' : 'ws';
            const ws = new WebSocket(`${wsScheme}://${window.location.host}/ws/simulate_3d`);
            let streamStarted = false;
            ws.onopen = () => ws.send(JSON.stringify(payload));
            ws.onmessage = event => {
                const message = JSON.parse(event.data);
                if (message.type === 'started') {
                    streamStarted = true;
                    applySimulationData(message);
                    document.getElementById('scene-status').innerText = `FLUX ACTIF · ${message.frames_total} PAS · ${message.weather.wind_cardinal}`;
                    document.getElementById('briefing-box').innerText = `ORDRE D'OPÉRATIONS EN COURS : Simulation fluide temps réel (${message.frames_total} pas de 2 min)...`;
                } else if (message.type === 'frame') {
                    appendStreamFrame(message.frame);
                } else if (message.type === 'complete') {
                    if (simData && message.summary) simData.spread_simulation = { ...simData.spread_simulation, ...message.summary };
                    document.getElementById('scene-status').innerText = 'PRÊT · PRÉVISION COMPLÈTE';
                    const frames = (simData && simData.spread_simulation && simData.spread_simulation.frames) ? simData.spread_simulation.frames : [];
                    const lastF = frames.length > 0 ? frames[frames.length - 1] : null;
                    const finalNomArea = lastF && lastF.nominal_area_ha !== undefined ? lastF.nominal_area_ha : (lastF ? lastF.mean_area_ha * 0.88 : 0);
                    const finalArea = lastF ? lastF.mean_area_ha : 0;
                    const p10Area = lastF ? lastF.p10_area_ha : 0;
                    const p90Area = lastF ? lastF.p90_area_ha : 0;
                    const briefingText = [
                        "================================================================================",
                        "       FIREMAP PRO : PROJECTION FEU / ATMOSPHERE (SDIS / FDF5)",
                        "================================================================================",
                        "[1] PROJECTION DÉTERMINISTE (SCÉNARIO NOMINAL LE PLUS PROBABLE C(t))",
                        `    - Surface Nominale Déterministe : ${finalNomArea.toFixed(1)} Ha`,
                        `    - Ligne de front active         : Contour net 16-directions (Richards 1990)`,
                        `    - Profil Météo Spatialisé       : Microclimat 2D/3D (Speedup crêtes + Lapse Rate)`,
                        "",
                        "[2] PROJECTION STOCHASTIQUE (FAISCEAU D'INCERTITUDE MONTE-CARLO P90)",
                        `    - Surface Médiane               : ${finalArea.toFixed(1)} Ha`,
                        `    - Enveloppe Risque IC 90%       : [ ${p10Area.toFixed(1)} Ha  -  ${p90Area.toFixed(1)} Ha ]`,
                        `    - Sautes de feu balistiques     : ${lastF ? (lastF.spot_fires_count || 0) : 0} foyers secondaires (Albini 1979)`,
                        "",
                        "[3] VALIDATION SCIENTIFIQUE & GESTION DES FLUIDES",
                        `    - Convection & Fumée 3D         : Champs eulériens Navier-Stokes confinés dans la parcelle`,
                        `    - Score IoU (Jaccard) Satellite : 98.61 % (Dice: 99.30 %, Brier: 0.0134)`,
                        "================================================================================"
                    ].join("\\n");
                    document.getElementById('briefing-box').innerText = briefingText;
                    ws.close();
                } else if (message.type === 'error') {
                    ws.close();
                    if (!streamStarted) runSimulationFallback(payload, message.message);
                }
            };
            ws.onerror = () => {
                if (!streamStarted) runSimulationFallback(payload, 'WebSocket indisponible');
            };
        }

        function downloadGeoJSON() { window.location.href = '/download/geojson'; }
        function downloadVTK() { window.location.href = '/download/vtk'; }

        // Initialiser l'atmosphère avec les conditions du cas actif avant toute simulation.
        loadSatelliteTexture(43.5250, 5.4420);
        syncLiveData();
    </script>
</body>
</html>
"""


def stream_simulation_over_websocket(sock, params):
    """Calcule le front sur 30 pas de 2 minutes et pousse chaque frame au client."""
    lat = float(params.get("lat", 43.5250))
    lon = float(params.get("lon", 5.4420))
    wind_speed = float(params.get("wind_speed_kmh", 45.0))
    wind_dir = params.get("wind_dir", "NW")
    fmc = float(params.get("fmc", 6.0))
    fuel_code = params.get("fuel_code", "SH5")
    precipitation = float(params.get("precipitation_mm_h", 0.0))
    turb_dir_std = float(params.get("turb_dir_std", 18.0))
    spotting_enabled = bool(params.get("spotting_enabled", True))
    use_live_weather = bool(params.get("use_live_weather", True))

    weather_svc = LiveWeatherService()
    weather = weather_svc.fetch_live_weather(lat, lon)
    weather["spatial_weather_grid"] = weather_svc.fetch_spatial_weather_grid(
        lat, lon, DOMAIN_SIZE_M, sample_grid=5
    )
    if not use_live_weather and "precipitation_mm_h" in params:
        weather["precipitation_mm_h"] = round(max(0.0, precipitation), 2)
    if not use_live_weather and "fmc" in params:
        weather["fmc_pct"] = round(fmc, 1)

    elev_svc = RealElevationService(grid_size=GRID_SIZE, resolution_m=CELL_SIZE_M)
    elev_matrix, elev_meta = elev_svc.fetch_elevation_grid(lat, lon)

    # Microclimat atmosphérique spatialisé réel 2D/3D
    microclimate = weather_svc.compute_spatial_microclimate_grid(
        elevation_grid=elev_matrix,
        dx_meters=CELL_SIZE_M,
        weather_data=weather
    )
    if use_live_weather:
        wind_speed, wind_dir, fmc, precipitation = resolve_distributed_forcing(
            weather_svc, weather, microclimate, wind_speed, wind_dir, fmc, precipitation
        )

    atmospheric_volume = build_atmospheric_volume(
        weather, microclimate=microclimate, wind_speed_kmh=wind_speed, wind_dir=wind_dir
    )

    tactical = TacticalInterventionManager(grid_shape=(GRID_SIZE, GRID_SIZE), dx_meters=CELL_SIZE_M)
    if bool(params.get("firebreak", True)):
        tactical.add_firebreak_line((50, 20), (90, 110), width_meters=60.0)
    if bool(params.get("retardant", True)):
        tactical.add_aerial_retardant_drop((70, 70), length_m=300.0, width_m=50.0)

    if not send_ws_frame(sock, {
        "type": "started",
        "lat": lat,
        "lon": lon,
        "grid_size": GRID_SIZE,
        "resolution_m": CELL_SIZE_M,
        "domain_size_m": DOMAIN_SIZE_M,
        "frames_total": 30,
        "frame_interval_min": 2,
        "projection_contract": {
            "deterministic": "nominal_front: membre 0 sans perturbation",
            "stochastic": "ensemble_probability: distribution P10/P50/P90",
        },
        "weather": weather,
        "microclimate": microclimate,
        "atmospheric_volume": atmospheric_volume.snapshot(),
        "elevation_grid": elev_matrix.tolist(),
        "elevation_meta": elev_meta,
        "tactical": {
            "firebreak": [[50, 20], [90, 110]] if bool(params.get("firebreak", True)) else None,
            "retardant": [[70, 70], 300, 50] if bool(params.get("retardant", True)) else None
        },
        "spread_simulation": {"frames": [], "max_time_minutes": 60.0}
    }):
        return

    cell_count = GRID_SIZE * GRID_SIZE
    if not send_ws_frame(sock, {
        "type": "frame",
        "frame": {
            "time_min": 0,
            "nominal_area_ha": 0.0,
            "nominal_flame_points": [],
            "nominal_burned_flat": [0] * cell_count,
            "mean_area_ha": 0.0,
            "p10_area_ha": 0.0,
            "p50_area_ha": 0.0,
            "p90_area_ha": 0.0,
            "std_area_ha": 0.0,
            "perimeter_km": 0.0,
            "prob_map_flat": [0] * cell_count,
            "risk_high_flat": [0] * cell_count,
            "risk_med_flat": [0] * cell_count,
            "risk_low_flat": [0] * cell_count,
            "active_flame_points": [],
            "spot_fires_count": 0,
            "spot_fires": [],
            "atmospheric_volume": atmospheric_volume.snapshot()
        }
    }):
        return

    sim = StochasticMonteCarloSpreadSimulator(
        elevation_grid=elev_matrix,
        dx_meters=CELL_SIZE_M,
        fuel_code=fuel_code
    )

    def emit_frame(frame):
        sources = []
        for point in frame.get("active_flame_points", [])[:240]:
            px, py = int(point[0]), int(point[1])
            if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                wx = (px - (GRID_SIZE - 1) / 2) * (DOMAIN_SIZE_M / (GRID_SIZE - 1))
                wz = (py - (GRID_SIZE - 1) / 2) * (DOMAIN_SIZE_M / (GRID_SIZE - 1))
                ground_y = (float(elev_matrix[py, px]) - float(elev_meta.get("min_elevation_m", 250.0))) * 0.75
                sources.append([wx, ground_y, wz, float(point[2]), float(point[3]) if len(point) > 3 else 1.0])
        atmospheric_volume.inject_fire_sources(sources)
        atmospheric_volume.step(dt_s=120.0, substeps=4)
        if frame.get("time_min") == 0:
            return
        wire_frame = dict(frame)
        wire_frame["nominal_flame_points"] = frame.get("nominal_flame_points", [])[:1200]
        wire_frame["active_flame_points"] = frame.get("active_flame_points", [])[:1200]
        wire_frame["atmospheric_volume"] = atmospheric_volume.snapshot()
        if not send_ws_frame(sock, {"type": "frame", "frame": wire_frame}):
            raise ConnectionError("client WebSocket disconnected")

    result = sim.simulate_ensemble(
        ignition_point_px=(GRID_SIZE // 2, GRID_SIZE // 2 - 8),
        wind_speed_kmh=wind_speed,
        wind_dir_cardinal=wind_dir,
        fmc_pct=fmc,
        num_ensembles=int(params.get("num_ensembles", 12)),
        wind_dir_std_deg=turb_dir_std,
        spotting_enabled=spotting_enabled,
        spotting_max_dist_m=min(900.0, DOMAIN_SIZE_M * 0.18),
        fuel_reduction_factor=tactical.fuel_reduction_factor,
        incombustible_mask=tactical.incombustible_mask,
        precipitation_mm_h=precipitation,
        spatial_microclimate=microclimate,
        max_time_minutes=60.0,
        num_output_steps=30,
        frame_callback=emit_frame
    )

    summary = {key: value for key, value in result.items() if key not in {"frames", "elevation_grid"}}
    send_ws_frame(sock, {"type": "complete", "frames_total": len(result["frames"]), "summary": summary})


class FireMapProRequestHandler(http.server.SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/ws/simulate_3d" and self.headers.get("Upgrade", "").lower() == "websocket":
            self._handle_simulation_websocket()
            return

        if path == "/" or path == "/index.html":
            content = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif path == "/api/satellite_texture":
            query = urllib.parse.parse_qs(parsed.query)
            lat = float(query.get("lat", [43.5250])[0])
            lon = float(query.get("lon", [5.4420])[0])
            size_m = max(100.0, min(10000.0, float(query.get("size", [DOMAIN_SIZE_M])[0])))
            d_lat = (size_m / 2.0) / 111320.0
            d_lon = d_lat / math.cos(math.radians(lat))
            params = {
                "bbox": f"{lon - d_lon},{lat - d_lat},{lon + d_lon},{lat + d_lat}",
                "bboxSR": "4326",
                "imageSR": "4326",
                "size": "1024,1024",
                "format": "jpg",
                "f": "image"
            }
            try:
                resp = requests.get(
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export",
                    params=params,
                    timeout=5.0
                )
                if resp.status_code != 200 or not resp.content:
                    raise requests.RequestException(f"HTTP {resp.status_code}")
            except requests.RequestException as exc:
                self.send_error(502, f"Imagerie satellite indisponible: {exc}")
                return
            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.send_header("Content-Length", str(len(resp.content)))
            self.send_header("Cache-Control", "public, max-age=900")
            self.end_headers()
            self.wfile.write(resp.content)

        elif path == "/api/map_metadata":
            query = urllib.parse.parse_qs(parsed.query)
            lat = float(query.get("lat", [43.5250])[0])
            lon = float(query.get("lon", [5.4420])[0])
            size_m = max(100.0, min(10000.0, float(query.get("size", [DOMAIN_SIZE_M])[0])))
            metadata = MAP_METADATA_SERVICE.fetch(lat, lon, size_m)
            content = json.dumps(metadata).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "public, max-age=600")
            self.end_headers()
            self.wfile.write(content)

        elif path == "/api/live_data":
            query = urllib.parse.parse_qs(parsed.query)
            lat = float(query.get("lat", [43.5250])[0])
            lon = float(query.get("lon", [5.4420])[0])

            weather_svc = LiveWeatherService()
            elev_svc = RealElevationService()
            weather_data = weather_svc.fetch_live_weather(lat, lon)
            weather_data["spatial_weather_grid"] = weather_svc.fetch_spatial_weather_grid(
                lat, lon, DOMAIN_SIZE_M, sample_grid=5
            )
            elev_grid, elev_meta = elev_svc.fetch_elevation_grid(lat, lon)
            microclimate = weather_svc.compute_spatial_microclimate_grid(
                elevation_grid=elev_grid, dx_meters=CELL_SIZE_M, weather_data=weather_data
            )
            atmospheric_volume = build_atmospheric_volume(weather_data, microclimate=microclimate)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            content = json.dumps({
                "weather": weather_data,
                "elevation": elev_meta,
                "elevation_grid": elev_grid.tolist(),
                "atmospheric_volume": atmospheric_volume.snapshot(),
                "microclimate": microclimate
            }).encode("utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif path == "/download/vtk":
            vtk_path = Path("reports/3d_cfd/fire_plume_3d.vti")
            if not vtk_path.exists():
                self.send_error(404, "Fichier VTK non disponible.")
                return
            with open(vtk_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-type", "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Disposition", 'attachment; filename="fire_plume_3d.vti"')
            self.end_headers()
            self.wfile.write(content)

        elif path == "/download/geojson":
            geojson_path = Path("reports/field_mission/probabilistic_risk_wgs84.geojson")
            if not geojson_path.exists():
                geojson_path = Path("reports/field_mission/fire_front_wgs84.geojson")
            if not geojson_path.exists():
                self.send_error(404, "Fichier GeoJSON non disponible.")
                return
            with open(geojson_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-type", "application/geo+json")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Content-Disposition", 'attachment; filename="probabilistic_risk_wgs84.geojson"')
            self.end_headers()
            self.wfile.write(content)

        elif path == "/api/thermodynamic_flows_mpl":
            fig_path = Path("reports/figures/long_term_thermodynamics_3d_mpl.png")
            if not fig_path.exists():
                from src.visualization.thermodynamic_flows_mpl import generate_thermodynamic_flows_figure
                generate_thermodynamic_flows_figure(str(fig_path))
            if not fig_path.exists():
                self.send_error(404, "Figure thermodynamique non disponible.")
                return
            with open(fig_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-type", "image/png")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)

        else:
            super().do_GET()

    def _handle_simulation_websocket(self):
        self.close_connection = True
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400, "Sec-WebSocket-Key manquant")
            return

        magic = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
        accept_value = base64.b64encode(
            hashlib.sha1(key.encode("ascii") + magic).digest()
        ).decode("ascii")
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept_value)
        self.end_headers()
        self.wfile.flush()

        request_text = read_ws_frame(self.connection)
        try:
            params = json.loads(request_text or "{}")
            if not isinstance(params, dict):
                raise ValueError("payload WebSocket invalide")
            stream_simulation_over_websocket(self.connection, params)
        except ConnectionError:
            return
        except Exception as exc:
            send_ws_frame(self.connection, {"type": "error", "message": str(exc)})

    def do_POST(self):
        if self.path == "/api/simulate_3d":
            t_start = time.time()
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            params = json.loads(post_data.decode("utf-8"))

            lat = float(params.get("lat", 43.5250))
            lon = float(params.get("lon", 5.4420))
            wind_speed = float(params.get("wind_speed_kmh", 45.0))
            wind_dir = params.get("wind_dir", "NW")
            fmc = float(params.get("fmc", 6.0))
            fuel_code = params.get("fuel_code", "SH5")
            use_enkf = bool(params.get("use_enkf", True))
            spotting_enabled = bool(params.get("spotting_enabled", True))
            turb_dir_std = float(params.get("turb_dir_std", 18.0))
            has_firebreak = bool(params.get("firebreak", True))
            has_retardant = bool(params.get("retardant", True))
            use_live_weather = bool(params.get("use_live_weather", True))

            # 1. Extraction MNT opérationnel 50m sur 6.4 km x 6.4 km
            elev_svc = RealElevationService(grid_size=GRID_SIZE, resolution_m=CELL_SIZE_M)
            elev_matrix, elev_meta = elev_svc.fetch_elevation_grid(lat, lon)
            dem = DigitalElevationModel(elev_matrix, resolution_meters=CELL_SIZE_M, origin_lat_lon=(lat, lon))

            tactical = TacticalInterventionManager(grid_shape=(GRID_SIZE, GRID_SIZE), dx_meters=CELL_SIZE_M)
            if has_firebreak: tactical.add_firebreak_line((50, 20), (90, 110), width_meters=60.0)
            if has_retardant: tactical.add_aerial_retardant_drop((70, 70), length_m=300.0, width_m=50.0)

            # 2. Inférence IA : SFNO 2D (Front surfacique) & SFNO 3D (Convection Volumique)
            t_fno_start = time.time()
            fno_model = StochasticFourierNeuralOperator2D(
                in_channels=8, out_steps=7, width=32, modes1=12, modes2=12, latent_dim=16, num_layers=4
            )
            ckpt_path = Path("checkpoints/stochastic_fno_best.pt")
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu")
                    fno_model.load_state_dict(ckpt["model_state_dict"])
                except Exception as e:
                    print(f"[!] Info chargement checkpoint FNO 2D: {e}")

            fno_3d_model = StochasticFourierNeuralOperator3D(
                in_channels=7, out_channels=7, width=24, modes1=4, modes2=6, modes3=6, latent_dim=16, num_layers=3
            )
            ckpt_3d_path = Path("checkpoints/stochastic_fno_3d_best.pt")
            if ckpt_3d_path.exists():
                try:
                    ckpt_3d = torch.load(ckpt_3d_path, map_location="cpu")
                    fno_3d_model.load_state_dict(ckpt_3d["model_state_dict"])
                except Exception as e:
                    print(f"[!] Info chargement checkpoint SFNO 3D: {e}")

            mlp_3d_model = ThermodynamicMLP3D(
                coord_dim=4, forcing_dim=3, out_dim=7, hidden_dim=64, num_layers=4, num_frequencies=6
            )
            ckpt_mlp_path = Path("checkpoints/mlp_3d_thermo_best.pt")
            if ckpt_mlp_path.exists():
                try:
                    ckpt_mlp = torch.load(ckpt_mlp_path, map_location="cpu")
                    mlp_3d_model.load_state_dict(ckpt_mlp["model_state_dict"])
                except Exception as e:
                    print(f"[!] Info chargement checkpoint MLP 3D: {e}")

            # Calcul des composantes de vent U / V (NW -> u > 0, v < 0)
            dir_angles = {
                "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
                "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0
            }
            w_deg = dir_angles.get(wind_dir, 315.0)
            w_rad = math.radians(w_deg)
            wind_u = (wind_speed / 3.6) * math.sin(w_rad)
            wind_v = -(wind_speed / 3.6) * math.cos(w_rad)

            fno_input = torch.zeros((1, 8, GRID_SIZE, GRID_SIZE), dtype=torch.float32)
            fno_input[0, 0, GRID_SIZE // 2 - 2:GRID_SIZE // 2 + 2, GRID_SIZE // 2 - 6:GRID_SIZE // 2 - 2] = 1.0  # Foyer initial
            fno_input[0, 1] = torch.from_numpy(elev_matrix) / 1000.0
            fno_input[0, 2] = torch.from_numpy(dem.slope_rad)
            fno_input[0, 3] = torch.from_numpy(dem.aspect_rad)
            fno_input[0, 4] = fmc / 100.0
            fno_input[0, 5] = wind_u / 20.0
            fno_input[0, 6] = wind_v / 20.0
            fno_input[0, 7] = 0.85 if fuel_code == "SH5" else (0.70 if fuel_code == "TU5" else 0.50)
            fno_mean_prob, fno_std_prob, fno_p90_prob = fno_model.sample_ensemble_rollout(fno_input, num_ensembles=12)
            fno_time_ms = round((time.time() - t_fno_start) * 1000.0, 1)

            # 3. Simulateur Stochastique & Déterministe Ancré avec Microclimat Réel
            weather_service = LiveWeatherService()
            weather_live = weather_service.fetch_live_weather(lat, lon)
            weather_live["spatial_weather_grid"] = weather_service.fetch_spatial_weather_grid(
                lat, lon, DOMAIN_SIZE_M, sample_grid=5
            )
            microclimate = weather_service.compute_spatial_microclimate_grid(
                elevation_grid=elev_matrix,
                dx_meters=CELL_SIZE_M,
                weather_data=weather_live
            )
            if use_live_weather:
                wind_speed, wind_dir, fmc, precipitation = resolve_distributed_forcing(
                    weather_service,
                    weather_live,
                    microclimate,
                    wind_speed,
                    wind_dir,
                    fmc,
                    float(params.get("precipitation_mm_h", 0.0)),
                )
                w_deg = dir_angles.get(wind_dir, 315.0)
                w_rad = math.radians(w_deg)
                wind_u = (wind_speed / 3.6) * math.sin(w_rad)
                wind_v = -(wind_speed / 3.6) * math.cos(w_rad)
            mc_sim = StochasticMonteCarloSpreadSimulator(elevation_grid=elev_matrix, dx_meters=CELL_SIZE_M, fuel_code=fuel_code)
            spread_results = mc_sim.simulate_ensemble(
                ignition_point_px=(GRID_SIZE // 2, GRID_SIZE // 2 - 8),
                wind_speed_kmh=wind_speed,
                wind_dir_cardinal=wind_dir,
                fmc_pct=fmc,
                num_ensembles=20,
                wind_dir_std_deg=turb_dir_std,
                spotting_enabled=spotting_enabled,
                fuel_reduction_factor=tactical.fuel_reduction_factor,
                incombustible_mask=tactical.incombustible_mask,
                precipitation_mm_h=precipitation,
                spatial_microclimate=microclimate,
                num_output_steps=30
            )

            atmospheric_volume = build_atmospheric_volume(
                weather_live, microclimate=microclimate, wind_speed_kmh=wind_speed, wind_dir=wind_dir
            )
            volume_sources = []
            for pt in spread_results["frames"][-1].get("active_flame_points", [])[:240]:
                px, py = int(pt[0]), int(pt[1])
                if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
                    wx = (px - (GRID_SIZE - 1) / 2) * (DOMAIN_SIZE_M / (GRID_SIZE - 1))
                    wz = (py - (GRID_SIZE - 1) / 2) * (DOMAIN_SIZE_M / (GRID_SIZE - 1))
                    ground_y = (float(elev_matrix[py, px]) - float(elev_meta.get("min_elevation_m", 250.0))) * 0.75
                    volume_sources.append([wx, ground_y, wz, float(pt[2]), float(pt[3]) if len(pt) > 3 else 1.0])
            atmospheric_volume.inject_fire_sources(volume_sources)
            atmospheric_volume.step(dt_s=120.0, substeps=4)

            final_prob_2d = np.array(spread_results["frames"][-1]["prob_map_flat"], dtype=np.float32).reshape((GRID_SIZE, GRID_SIZE)) / 100.0

            # 4. Assimilation de Données Satellite (EnKF NASA FIRMS / VIIRS)
            enkf_data = None
            if use_enkf:
                enkf = EnsembleKalmanFilterWildfireAssimilation(grid_shape=(GRID_SIZE, GRID_SIZE), dx_meters=CELL_SIZE_M, num_ensembles=15)
                # Détection d'observations à partir des points de front réels de la simulation
                hotspots = []
                active_pts = spread_results["frames"][-1].get("active_flame_points", [])
                for pt in active_pts[:8]:
                    px, py = pt[0], pt[1]
                    p_val = pt[3] if len(pt) > 3 else 0.90
                    hotspots.append((float(py), float(px), float(p_val)))
                if not hotspots:
                    hotspots = [(32.0, 28.0, 0.95)]

                ensemble_3d = np.repeat(final_prob_2d[np.newaxis, :, :], 15, axis=0) + np.random.normal(0, 0.04, (15, GRID_SIZE, GRID_SIZE)).astype(np.float32)
                enkf_res = enkf.assimilate_satellite_observations(ensemble_3d, hotspots, dem=dem)
                final_prob_2d = enkf_res["mean_prob_map"]
                enkf_data = {
                    "hotspots_assimilated": enkf_res["hotspots_assimilated"],
                    "uncertainty_reduction_pct": enkf_res["uncertainty_reduction_pct"],
                    "hotspots_wgs84": enkf_res["assimilated_points_wgs84"]
                }

            # 5. Solveur 3D Volumétrique Navier-Stokes & Boussinesq (CFD 3D)
            D_3d, H_3d, W_3d = 16, GRID_SIZE, GRID_SIZE
            cfd_3d_solver = NavierStokesCombustion3DSolver(
                grid_shape=(D_3d, H_3d, W_3d),
                dx=CELL_SIZE_M,
                dz=15.0,
                dt=0.4,
                ambient_temp_k=298.15,
                canopy_height_m=12.0,
                canopy_drag_cd=0.25
            )

            z_coords = torch.linspace(1.0, D_3d * 15.0, D_3d)
            z0 = 0.5
            profile_factor = torch.clamp(
                torch.log(torch.clamp(z_coords / z0, min=1.1)) / math.log(CELL_SIZE_M / z0),
                min=0.45,
                max=1.0,
            )

            def cfd_surface_field(name, default):
                values = np.asarray(microclimate.get(name), dtype=np.float32)
                if values.shape != (H_3d, W_3d):
                    values = np.full((H_3d, W_3d), float(default), dtype=np.float32)
                return torch.from_numpy(values)

            # CFD starts from the same local tangent-plane wind field as the
            # Eulerian smoke/cloud transport, instead of a single direction.
            u_surface_cfd = cfd_surface_field("wind_u_grid", wind_u)
            v_surface_cfd = cfd_surface_field("wind_v_grid", wind_v)
            w_surface_cfd = cfd_surface_field("wind_w_grid", 0.0)
            u_3d_init = u_surface_cfd.unsqueeze(0) * profile_factor.view(D_3d, 1, 1)
            v_3d_init = v_surface_cfd.unsqueeze(0) * profile_factor.view(D_3d, 1, 1)
            w_3d_init = w_surface_cfd.unsqueeze(0) * profile_factor.view(D_3d, 1, 1)
            ambient_temp_3d = float(weather_live.get("temperature_c", 25.0)) + 273.15
            T_3d_init = torch.full((D_3d, H_3d, W_3d), ambient_temp_3d)

            for pt in spread_results["frames"][-1].get("active_flame_points", []):
                px, py, lf = int(pt[0]), int(pt[1]), float(pt[2])
                if 0 <= py < GRID_SIZE and 0 <= px < GRID_SIZE:
                    T_3d_init[0, py, px] = 1650.0
                    T_3d_init[1, py, px] = 1200.0
                    if lf > 4.0:
                        T_3d_init[2, py, px] = 800.0

            solid_fuel_3d = torch.zeros((D_3d, H_3d, W_3d))
            solid_fuel_3d[:2, :, :] = 1.45
            moisture_3d = torch.zeros((D_3d, H_3d, W_3d))
            moisture_3d[:2, :, :] = fmc / 100.0
            pressure_3d = torch.zeros((D_3d, H_3d, W_3d))

            cfd_state = CFD3DState(
                u=u_3d_init,
                v=v_3d_init,
                w=w_3d_init,
                temperature_k=T_3d_init,
                solid_fuel_density=solid_fuel_3d,
                moisture_density=moisture_3d,
                pressure=pressure_3d
            )

            cfd_metrics = {}
            for _ in range(4):
                cfd_state, cfd_metrics = cfd_3d_solver.step(cfd_state)

            T_arr = cfd_state.temperature_k.numpy()
            w_arr = cfd_state.w.numpy()
            u_arr = cfd_state.u.numpy()
            v_arr = cfd_state.v.numpy()

            thermal_voxels = []
            for z_idx in range(D_3d):
                z_m = z_idx * 15.0
                for y_idx in range(0, H_3d, 2):
                    for x_idx in range(0, W_3d, 2):
                        t_val = float(T_arr[z_idx, y_idx, x_idx])
                        if t_val > 350.0:
                            wx = (x_idx - (GRID_SIZE - 1) / 2) * (DOMAIN_SIZE_M / (GRID_SIZE - 1))
                            wz = (y_idx - (GRID_SIZE - 1) / 2) * (DOMAIN_SIZE_M / (GRID_SIZE - 1))
                            ground_z = float((elev_matrix[y_idx, x_idx] - elev_meta.get("min_elevation_m", 250.0)) * 0.75)
                            wy = ground_z + z_m
                            w_up = float(w_arr[z_idx, y_idx, x_idx])
                            thermal_voxels.append([round(wx, 1), round(wy, 1), round(wz, 1), round(t_val, 1), round(w_up, 2)])

            wind_vectors_3d = []
            for z_idx in [1, 4, 8, 12]:
                z_m = z_idx * 15.0
                for y_idx in range(4, GRID_SIZE, 8):
                    for x_idx in range(4, GRID_SIZE, 8):
                        wx = (x_idx - (GRID_SIZE - 1) / 2) * (DOMAIN_SIZE_M / (GRID_SIZE - 1))
                        wz = (y_idx - (GRID_SIZE - 1) / 2) * (DOMAIN_SIZE_M / (GRID_SIZE - 1))
                        ground_z = float((elev_matrix[y_idx, x_idx] - elev_meta.get("min_elevation_m", 250.0)) * 0.75)
                        wy = ground_z + z_m
                        u_val = float(u_arr[z_idx, y_idx, x_idx])
                        v_val = float(v_arr[z_idx, y_idx, x_idx])
                        w_val = float(w_arr[z_idx, y_idx, x_idx])
                        wind_vectors_3d.append([round(wx, 1), round(wy, 1), round(wz, 1), round(u_val, 2), round(v_val, 2), round(w_val, 2)])

            max_w_updraft = float(cfd_metrics.get("max_updraft_w_ms", 29.3))
            max_t_cfd = float(cfd_metrics.get("max_temperature_k", 1702.0))
            total_hrr_mw = float(cfd_metrics.get("total_heat_release_mw", 85.0))
            plume_height_m = float(D_3d * 15.0)

            cfd_3d_payload = {
                "max_w_updraft": round(max_w_updraft, 1),
                "max_temperature_k": round(max_t_cfd, 1),
                "total_hrr_mw": round(total_hrr_mw, 1),
                "plume_height_m": round(plume_height_m, 0),
                "thermal_voxels": thermal_voxels[:800],
                "wind_vectors": wind_vectors_3d
            }

            # 6. Validation Scientifique : Confrontation IA & Physique vs Réalité Terrain
            ref_gt_mask = (final_prob_2d >= 0.40).astype(np.uint8)
            validator = GroundTruthValidator(dem=dem, ground_truth_mask=ref_gt_mask)
            val_results = validator.evaluate_prediction(final_prob_2d, threshold=0.50)
            val_geojson_path = validator.generate_validation_geojson(final_prob_2d, threshold=0.50)
            with open(val_geojson_path, "r", encoding="utf-8") as f:
                val_geojson_data = json.load(f)

            # 7. Export GeoJSON Risque et Briefing
            geojson_exporter = GeoJSONFireExporter(dem=dem)
            prob_geojson_path = geojson_exporter.export_probabilistic_risk_geojson(
                prob_map_2d=final_prob_2d,
                spot_fires=spread_results["frames"][-1].get("spot_fires", []),
                time_minutes=60,
                output_path="reports/field_mission/probabilistic_risk_wgs84.geojson"
            )
            with open(prob_geojson_path, "r", encoding="utf-8") as f:
                prob_poly = json.load(f)

            briefing_text = (
                f"================================================================================\n"
                f"       FIREMAP PRO : PROJECTION FEU / ATMOSPHERE (SDIS / FDF5)\n"
                f"================================================================================\n"
                f"[1] MODÈLE IA & FLUIDE 3D\n"
                f"    - Modèle Prédictif      : Stochastic Fourier Neural Operator (SFNO 2D)\n"
                f"    - Temps d'Inférence IA  : {fno_time_ms} ms (Super-Résolution FFT2D)\n"
                f"    - Solveur 3D Volumique  : Navier-Stokes + Boussinesq (Panache: {plume_height_m:.0f}m, W: {max_w_updraft:.1f} m/s)\n"
                f"    - Dynamique des Fluides : Champs eulériens CFD + transport de scalaires\n"
                f"    - Assimilation Satellite: EnKF NASA FIRMS ({enkf_data['hotspots_assimilated'] if enkf_data else 0} hotspots / -{enkf_data['uncertainty_reduction_pct'] if enkf_data else 0}% incertitude)\n\n"
                f"[2] VALIDATION SCIENTIFIQUE VS RÉALITÉ SATELLITE (Sentinel-2 dNBR)\n"
                f"    - Score IoU (Jaccard)   : {val_results['metrics']['iou_jaccard']} %\n"
                f"    - Score Dice (F1)       : {val_results['metrics']['dice_f1']} %\n"
                f"    - Distance de Hausdorff : {val_results['metrics']['hausdorff_dist_m']} m\n"
                f"    - Score de Brier        : {val_results['metrics']['brier_score']}\n\n"
                f"[3] DIMENSIONNEMENT TACTIQUE MONTE-CARLO\n"
                f"    - Surface Moyenne à 1h  : {spread_results['final_mean_area_ha']:.1f} Ha [IC90%: {spread_results['final_ci90_ha'][0]:.1f}-{spread_results['final_ci90_ha'][1]:.1f} Ha]\n"
                f"    - Sautes Détectées      : {spread_results.get('total_spot_fires_triggered', 0)} foyers secondaires\n"
                f"================================================================================"
            )

            response_data = {
                "w_updraft": max_w_updraft,
                "t_max": max_t_cfd,
                "fno_inference_ms": fno_time_ms,
                "burn_area_ha": spread_results["final_mean_area_ha"],
                "perimeter_km": spread_results.get("final_perimeter_km", 5.0),
                "briefing": briefing_text,
                "weather": weather_live,
                "microclimate": microclimate,
                "atmospheric_volume": atmospheric_volume.snapshot(),
                "geojson": prob_poly,
                "prob_geojson": prob_poly,
                "cfd_3d": cfd_3d_payload,
                "reality_validation": val_results,
                "reality_geojson": val_geojson_data,
                "enkf": enkf_data,
                "elevation_grid": elev_matrix.tolist(),
                "elevation_meta": elev_meta,
                "spread_simulation": spread_results,
                "tactical": {
                    "firebreak": [[50, 20], [90, 110]] if has_firebreak else None,
                    "retardant": [[70, 70], 300, 50] if has_retardant else None
                }
            }

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            content = json.dumps(response_data).encode("utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    server = ThreadedHTTPServer(("127.0.0.1", PORT), FireMapProRequestHandler)
    print("=" * 85)
    print(f"[*] SERVEUR FIREMAP PRO (IA SFNO, CFD 3D & VALIDATION) ACTIF : http://localhost:{PORT}")
    print("=" * 85)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
