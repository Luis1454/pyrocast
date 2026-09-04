"""
run_3d_cfd_simulation.py
------------------------
Exécute la simulation 3D complète et authentique Navier-Stokes / Thermochimie couplée :
1. Initialisation volumique 3D [Z=16 couches, Y=64, X=64] avec profil logarithmique de vent atmosphérique
2. Résolution pas-à-pas des EDP 3D (Navier-Stokes, Boussinesq, Arrhenius, Évaporation, Stefan-Boltzmann)
3. Construction de l'Octree 3D volumique et calcul de compression mémoire
4. Exportation du fichier 3D VTK (.vti) pour ParaView / CAO
5. Génération des rendus 3D scientifiques (Coupes X-Z, Isovolume 3D, Maillage Octree)
"""

import sys
from pathlib import Path
import numpy as np
import torch

from src.physics.cfd_3d_engine import NavierStokesCombustion3DSolver, CFD3DState
from src.mesh.octree_3d_volumetric import Volumetric3DOctreeBuilder
from src.operational.vtk_exporter import VTK3DExporter
from src.visualization.visualizer_3d import Volumetric3DVisualizer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 85)
    print(" [*] FIREMAP HPC : SIMULATION 3D DIRECTE DES EQUATIONS DE NAVIER-STOKES & COMBUSTION")
    print(f" [*] Dispositif de calcul : {device}")
    print("=" * 85)

    # 1. Dimensions du domaine tridimensionnel
    D, H, W = 16, 64, 64  # Z (vertical), Y (sud-nord), X (ouest-est)
    dx = 20.0             # Résolution horizontale : 20m (Domaine 1280m x 1280m)
    dz = 15.0             # Résolution verticale : 15m (Élévation 240m)
    dt = 0.4              # Pas de temps CFL

    solver = NavierStokesCombustion3DSolver(
        grid_shape=(D, H, W),
        dx=dx,
        dz=dz,
        dt=dt,
        ambient_temp_k=298.15,
        canopy_height_m=15.0,
        canopy_drag_cd=0.25
    )

    # 2. Conditions initiales volumiques 3D
    # Profil logarithmique du vent atmosphérique en couche limite : U(z) = U_ref * ln(z/z0) / ln(z_ref/z0)
    z_coords = torch.linspace(1.0, D * dz, D, device=device)
    z0 = 0.5   # Rugosité de canopée
    u_ref = 12.0 # 12 m/s (~43 km/h) à z=30m
    u_profile = u_ref * torch.log(torch.clamp(z_coords / z0, min=1.1)) / np.log(30.0 / z0)

    u_3d = u_profile.view(D, 1, 1).expand(D, H, W).clone()
    v_3d = torch.zeros((D, H, W), device=device)
    w_3d = torch.zeros((D, H, W), device=device)

    # Champ de température initial : T_ambiant partout, sauf foyer au sol (z=0,1)
    T_3d = torch.full((D, H, W), 298.15, device=device)
    
    # Foyer initial intense au sol (X=300m, Y=640m)
    x_grid, y_grid = torch.meshgrid(torch.linspace(0, (W-1)*dx, W, device=device), torch.linspace(0, (H-1)*dx, H, device=device), indexing="xy")
    dist_ign = torch.sqrt((x_grid - 320.0)**2 + (y_grid - 640.0)**2)
    flame_ground = torch.exp(-(dist_ign**2) / (60.0**2)) * 1400.0 # +1400K -> 1700K

    T_3d[0] += flame_ground
    T_3d[1] += flame_ground * 0.6

    # Densité de combustible sec et humidité
    solid_fuel_3d = torch.zeros((D, H, W), device=device)
    solid_fuel_3d[:2, :, :] = 1.25  # 1.25 kg/m^3 de biomasse dans la canopée
    
    moisture_3d = torch.zeros((D, H, W), device=device)
    moisture_3d[:2, :, :] = 0.10    # 10% d'humidité

    # Pression initiale
    pressure_3d = torch.zeros((D, H, W), device=device)

    state = CFD3DState(
        u=u_3d,
        v=v_3d,
        w=w_3d,
        temperature_k=T_3d,
        solid_fuel_density=solid_fuel_3d,
        moisture_density=moisture_3d,
        pressure=pressure_3d
    )

    print(f"[*] Initialisation 3D terminee : Domaine {W*dx:.0f}m x {H*dx:.0f}m x {D*dz:.0f}m ({D*H*W} voxels)")
    print(f"[*] Profil de vent 3D : Logarithmique (U_sol = {u_profile[0]:.1f} m/s -> U_sommet = {u_profile[-1]:.1f} m/s)")
    print("\n--- AVANCEE TEMPORELLE DU SOLVEUR CFD 3D (NAVIER-STOKES + BOUSSINESQ) ---")

    # 3. Boucle d'intégration temporelle des EDP 3D
    num_steps = 15
    for step in range(1, num_steps + 1):
        state, metrics = solver.step(state)
        if step % 3 == 0 or step == num_steps:
            print(f"  Pas {step:02d}/{num_steps:02d} (t={step*dt:4.1f}s) | T_max: {metrics['max_temperature_k']:6.1f} K | W_updraft: {metrics['max_updraft_w_ms']:5.2f} m/s | Puissance: {metrics['total_heat_release_mw']:6.1f} MW | div(u): {metrics['mean_divergence']:.2e}")

    # 4. Construction de l'Octree 3D adaptatif
    print("\n[*] Generation de l'Octree 3D volumique...")
    # Tenseur d'état [7, D, H, W]
    state_tensor_7c = torch.zeros((7, D, H, W), device=device)
    state_tensor_7c[0] = state.temperature_k
    state_tensor_7c[1] = state.pressure
    state_tensor_7c[2] = state.u
    state_tensor_7c[3] = state.v
    state_tensor_7c[4] = state.w
    state_tensor_7c[5] = state.moisture_density
    state_tensor_7c[6] = state.solid_fuel_density

    octree_builder = Volumetric3DOctreeBuilder(
        domain_size_m=(D * dz, H * dx, W * dx),
        max_depth=3,
        min_depth=1,
        grad_threshold=0.30
    )
    mesh_3d = octree_builder.build_from_3d_state(state_tensor_7c)

    print(f"  Points cartesiens 3D denses : {D * H * W}")
    print(f"  Noeuds actifs dans l'Octree 3D: {mesh_3d.node_coords.shape[0]}")
    print(f"  Taux de compression memoire   : {mesh_3d.compression_rate_pct:.2f}%")
    print(f"  Aretes 3D Near-Field          : {mesh_3d.edge_index_near.shape[1]}")
    print(f"  Aretes 3D Far-Field (FMM)     : {mesh_3d.edge_index_far.shape[1]}")

    # 5. Exportation 3D VTK (.vti)
    print("\n[*] Exportation du tenseur volumique 3D au format VTK (.vti)...")
    vtk_exporter = VTK3DExporter(spacing=(dx, dx, dz))
    vtk_file = vtk_exporter.export_vti(
        temperature_3d=state.temperature_k,
        u_3d=state.u,
        v_3d=state.v,
        w_3d=state.w,
        fuel_3d=state.solid_fuel_density,
        output_path="reports/3d_cfd/fire_plume_3d.vti"
    )
    print(f"  [OK] Fichier VTK genere : {vtk_file}")

    # 6. Rendu des visualisations 3D scientifiques
    print("\n[*] Generation des diagnostics graphiques 3D...")
    viz_3d = Volumetric3DVisualizer(output_dir="reports/3d_cfd")
    
    p1 = viz_3d.plot_3d_orthogonal_slices(state.temperature_k, state.u, state.w, dx=dx, dz=dz)
    p2 = viz_3d.plot_3d_volumetric_plume(state.temperature_k, state.u, state.v, state.w, dx=dx, dz=dz)
    p3 = viz_3d.plot_3d_octree_mesh(mesh_3d.node_coords, mesh_3d.node_depths)

    print(f"  [OK] Coupe orthogonale panache X-Z : {p1}")
    print(f"  [OK] Rendu 3D Isovolume & Flottabilite: {p2}")
    print(f"  [OK] Maillage adaptatif Octree 3D  : {p3}")

    print("\n" + "=" * 85)
    print(" [OK] SIMULATION 3D EXACTE ET AUTHENTIQUE TERMINEE AVEC SUCCES.")
    print("=" * 85)


if __name__ == "__main__":
    main()
