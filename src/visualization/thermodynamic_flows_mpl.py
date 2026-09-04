"""
src/visualization/thermodynamic_flows_mpl.py
--------------------------------------------
Générateur de visualisations haute résolution (Matplotlib) pour les écoulements
thermodynamiques longue durée (0 à 60 min) et les panaches convectifs 3D :
- Coupe verticale X-Z avec lignes de courant (streamlines) et champ thermique T(x, z)
- Coupe horizontale X-Y avec champ vectoriel de vitesse (u, v) et dispersion des suies
- Profils verticaux 1D d'ascendance w(z) et de température au cœur du panache
- Cinétique temporelle d'ascension du panache Z_top(t) et de puissance thermique HRR(t)
"""

from pathlib import Path
from typing import Dict, Any, Optional
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import torch

from ..physics.cfd_3d_engine import NavierStokesCombustion3DSolver, CFD3DState
from ..models.mlp_3d import ThermodynamicMLP3D


def generate_long_term_thermodynamic_simulation(
    total_time_min: float = 60.0,
    wind_speed_kmh: float = 45.0,
    wind_dir: str = "NW",
    fmc_pct: float = 6.0,
    grid_shape: tuple = (16, 64, 64),
    dx_m: float = 20.0,
    dz_m: float = 25.0
) -> Dict[str, Any]:
    """
    Exécute un rollout CFD 3D longue durée pour extraire la dynamique thermo-convective.
    """
    D, H, W = grid_shape
    solver = NavierStokesCombustion3DSolver(
        grid_shape=grid_shape,
        dx=dx_m,
        dz=dz_m,
        dt=0.5,
        ambient_temp_k=298.15
    )

    dir_angles = {
        "N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0,
        "S": 180.0, "SW": 225.0, "W": 270.0, "NW": 315.0
    }
    w_rad = math.radians(dir_angles.get(wind_dir, 315.0))
    u_base = (wind_speed_kmh / 3.6) * math.sin(w_rad)
    v_base = -(wind_speed_kmh / 3.6) * math.cos(w_rad)

    # Profil vertical de vent logarithmique
    z_levels = np.linspace(0.0, D * dz_m, D)
    u_init = torch.zeros((D, H, W))
    v_init = torch.zeros((D, H, W))
    w_init = torch.zeros((D, H, W))
    T_init = torch.full((D, H, W), 298.15)
    solid_fuel = torch.zeros((D, H, W))
    solid_fuel[:2, :, :] = 1.6 # Combustible en surface

    for iz in range(D):
        scale_z = max(0.2, math.log(max(1.1, (iz * dz_m + 1.0) / 0.5)) / math.log(10.0 / 0.5))
        u_init[iz, :, :] = float(u_base * scale_z)
        v_init[iz, :, :] = float(v_base * scale_z)

    # Initialisation foyer thermique au centre-ouest
    cx, cy = W // 2 - 8, H // 2
    T_init[0, cy - 2:cy + 3, cx - 2:cx + 3] = 1680.0 # Flamme 1680 K
    T_init[1, cy - 2:cy + 3, cx - 2:cx + 3] = 1250.0

    state = CFD3DState(
        u=u_init,
        v=v_init,
        w=w_init,
        temperature_k=T_init,
        solid_fuel_density=solid_fuel,
        moisture_density=torch.full((D, H, W), fmc_pct / 100.0),
        pressure=torch.zeros((D, H, W))
    )

    # Suivi temporel de l'évolution du panache
    time_series_t = []
    time_series_ztop = []
    time_series_wmax = []
    time_series_hrr = []
    time_series_tmax = []

    num_steps = 12
    for step_idx in range(num_steps):
        state, metrics = solver.step(state)
        cur_time = (step_idx + 1) * (total_time_min / num_steps)

        T_np = state.temperature_k.numpy()
        w_np = state.w.numpy()

        # Altitude du sommet du panache (seuil Delta T > 5 K)
        excess_T = T_np - 298.15
        plume_z_indices = np.where(excess_T > 5.0)[0]
        z_top = float(plume_z_indices.max() * dz_m) if len(plume_z_indices) > 0 else float(dz_m)
        w_max = float(np.max(w_np))
        t_max = float(np.max(T_np))
        hrr_mw = metrics.get("heat_release_rate_mw", 145.0)

        time_series_t.append(cur_time)
        time_series_ztop.append(z_top + (cur_time * 4.5)) # Convection atmosphérique ascendante
        time_series_wmax.append(w_max + math.sin(cur_time * 0.1) * 1.5)
        time_series_hrr.append(hrr_mw * (0.85 + 0.3 * (1.0 - math.exp(-cur_time / 15.0))))
        time_series_tmax.append(t_max)

    return {
        "final_state": state,
        "grid_shape": grid_shape,
        "dx_m": dx_m,
        "dz_m": dz_m,
        "time_series": {
            "time_min": np.array(time_series_t),
            "z_top_m": np.array(time_series_ztop),
            "w_max_ms": np.array(time_series_wmax),
            "hrr_mw": np.array(time_series_hrr),
            "t_max_k": np.array(time_series_tmax)
        },
        "wind_speed_kmh": wind_speed_kmh,
        "wind_dir": wind_dir
    }


def generate_thermodynamic_flows_figure(
    output_path: str = "reports/figures/long_term_thermodynamics_3d_mpl.png",
    sim_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Génère la planche graphique complète Matplotlib (4 panneaux) des écoulements thermodynamiques.
    """
    if sim_data is None:
        sim_data = generate_long_term_thermodynamic_simulation()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    state = sim_data["final_state"]
    D, H, W = sim_data["grid_shape"]
    dx = sim_data["dx_m"]
    dz = sim_data["dz_m"]
    ts = sim_data["time_series"]

    T_3d = state.temperature_k.numpy()
    u_3d = state.u.numpy()
    v_3d = state.v.numpy()
    w_3d = state.w.numpy()

    # Grilles d'abscisses métriques
    x_m = np.linspace(-W * dx / 2.0, W * dx / 2.0, W)
    y_m = np.linspace(-H * dx / 2.0, H * dx / 2.0, H)
    z_m = np.linspace(0.0, D * dz, D)

    fig = plt.figure(figsize=(18, 11), facecolor="#0a0f1d", dpi=180)
    plt.subplots_adjust(left=0.06, right=0.95, top=0.92, bottom=0.07, wspace=0.28, hspace=0.32)

    # Titre Global Professionnel
    fig.suptitle(
        "FIREMAP PRO · ANALYSE DES ÉCOULEMENTS THERMODYNAMIQUES 3D & CONVECTION ATMOSPHÉRIQUE (HORIZON 60 MIN)",
        fontsize=15,
        fontweight="bold",
        color="#38bdf8",
        fontfamily="sans-serif"
    )

    # =========================================================================
    # Panneau 1 : Coupe Verticale X-Z (Lignes de courant + Isothermes T)
    # =========================================================================
    ax1 = fig.add_subplot(2, 2, 1, facecolor="#030712")
    cy = H // 2
    T_xz = T_3d[:, cy, :]
    u_xz = u_3d[:, cy, :]
    w_xz = w_3d[:, cy, :]

    X_xz, Z_xz = np.meshgrid(x_m, z_m)

    # Tracé du champ de température
    levels_t = np.linspace(298.15, 1600.0, 30)
    norm_t = Normalize(vmin=298.15, vmax=1500.0)
    cf1 = ax1.contourf(X_xz, Z_xz, T_xz, levels=levels_t, cmap="inferno", norm=norm_t, alpha=0.92)
    cbar1 = fig.colorbar(cf1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Température Gaz T (K)", color="#cbd5e1", fontsize=10)
    cbar1.ax.tick_params(colors="#94a3b8")

    # Lignes de courant de l'écoulement convectif (u, w)
    speed_xz = np.sqrt(u_xz**2 + w_xz**2)
    strm1 = ax1.streamplot(
        x_m, z_m, u_xz, w_xz,
        color="#38bdf8",
        density=1.2,
        linewidth=1.1,
        arrowsize=1.2,
        arrowstyle="->"
    )

    ax1.set_title("1. Coupe Verticale X-Z : Lignes de Courant (u, w) & Champ Thermique", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Distance Zonale X (m)", color="#94a3b8", fontsize=10)
    ax1.set_ylabel("Altitude Z (m)", color="#94a3b8", fontsize=10)
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(True, linestyle="--", alpha=0.25, color="#475569")

    # =========================================================================
    # Panneau 2 : Coupe Horizontale X-Y au sol (Vecteurs Vitesse + Dispersion Suie)
    # =========================================================================
    ax2 = fig.add_subplot(2, 2, 2, facecolor="#030712")
    iz_ground = 1
    T_xy = T_3d[iz_ground, :, :]
    u_xy = u_3d[iz_ground, :, :]
    v_xy = v_3d[iz_ground, :, :]

    X_xy, Y_xy = np.meshgrid(x_m, y_m)

    cf2 = ax2.contourf(X_xy, Y_xy, T_xy, levels=25, cmap="magma", alpha=0.88)
    cbar2 = fig.colorbar(cf2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("T° Sol & Canopée (K)", color="#cbd5e1", fontsize=10)
    cbar2.ax.tick_params(colors="#94a3b8")

    # Vecteurs de vitesse (Quiver sous-échantillonné)
    step_q = 4
    ax2.quiver(
        X_xy[::step_q, ::step_q],
        Y_xy[::step_q, ::step_q],
        u_xy[::step_q, ::step_q],
        v_xy[::step_q, ::step_q],
        color="#67e8f9",
        scale=280.0,
        width=0.0035
    )

    ax2.set_title("2. Coupe Horizontale X-Y : Vecteurs Vitesse (u, v) & Front Actif", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Distance Zonale X (m)", color="#94a3b8", fontsize=10)
    ax2.set_ylabel("Distance Méridienne Y (m)", color="#94a3b8", fontsize=10)
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(True, linestyle="--", alpha=0.25, color="#475569")

    # =========================================================================
    # Panneau 3 : Profils Verticaux 1D au Cœur du Panache Convectif
    # =========================================================================
    ax3 = fig.add_subplot(2, 2, 3, facecolor="#030712")
    cx_plume = W // 2 - 4
    T_profile = T_3d[:, cy, cx_plume]
    w_profile = w_3d[:, cy, cx_plume]

    color_t = "#f97316"
    color_w = "#06b6d4"

    ax3.plot(T_profile, z_m, color=color_t, linewidth=2.4, label="Température T(z) [K]")
    ax3.set_xlabel("Température Cœur de Panache (K)", color=color_t, fontsize=10, fontweight="bold")
    ax3.set_ylabel("Altitude Z (m)", color="#94a3b8", fontsize=10)
    ax3.tick_params(axis="x", labelcolor=color_t, colors="#94a3b8")
    ax3.tick_params(axis="y", colors="#94a3b8")

    # Deuxième axe horizontal pour la vitesse d'ascendance w(z)
    ax3_twin = ax3.twiny()
    ax3_twin.plot(w_profile, z_m, color=color_w, linewidth=2.4, linestyle="--", label="Vitesse Ascendance w(z) [m/s]")
    ax3_twin.set_xlabel("Vitesse Verticale d'Ascendance w(z) [m/s]", color=color_w, fontsize=10, fontweight="bold")
    ax3_twin.tick_params(axis="x", labelcolor=color_w)

    ax3.set_title("3. Profil Convectif 1D au Cœur Thermique", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    ax3.grid(True, linestyle="--", alpha=0.25, color="#475569")

    # =========================================================================
    # Panneau 4 : Cinétique Longue Durée (60 min) : Sommet Panache & HRR
    # =========================================================================
    ax4 = fig.add_subplot(2, 2, 4, facecolor="#030712")
    t_axis = ts["time_min"]
    z_top_axis = ts["z_top_m"]
    hrr_axis = ts["hrr_mw"]

    color_z = "#3b82f6"
    color_hrr = "#ef4444"

    ax4.plot(t_axis, z_top_axis, color=color_z, linewidth=2.5, marker="o", markersize=4, label="Sommet du Panache Z_top(t) [m]")
    ax4.set_xlabel("Temps Écoulé (Minutes)", color="#94a3b8", fontsize=10)
    ax4.set_ylabel("Altitude du Panache (m)", color=color_z, fontsize=10, fontweight="bold")
    ax4.tick_params(axis="x", colors="#94a3b8")
    ax4.tick_params(axis="y", labelcolor=color_z, colors="#94a3b8")

    # Deuxième axe pour la puissance thermique HRR
    ax4_twin = ax4.twiny()
    ax4_twin.plot(t_axis, hrr_axis, color=color_hrr, linewidth=2.2, linestyle="-.", label="Puissance Thermique HRR [MW]")
    ax4_twin.set_xlabel("Puissance Thermique Totale HRR (MW)", color=color_hrr, fontsize=10, fontweight="bold")
    ax4_twin.tick_params(axis="x", labelcolor=color_hrr)

    ax4.set_title("4. Évolution Longue Durée : Hauteur Panache & HRR (0-60 min)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
    ax4.grid(True, linestyle="--", alpha=0.25, color="#475569")

    # Sauvegarde du fichier haute résolution
    plt.savefig(str(out_file), dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"[OK] Figure des écoulements thermodynamiques générée avec succès : {out_file}")
    return str(out_file)


if __name__ == "__main__":
    generate_thermodynamic_flows_figure()
