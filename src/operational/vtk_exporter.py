"""
src/operational/vtk_exporter.py
-------------------------------
Exportation volumique 3D au format standard scientifique VTK (Visualization Toolkit / ParaView) :
- Fichiers .vti (VTK XML ImageData) pour l'inspection 3D de champs continus
- Export des tenseurs complets : Température 3D, Vecteur Vitesse 3D (u,v,w), Densité de carburant, Pression
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple
import numpy as np
import torch


class VTK3DExporter:
    """
    Exportateur de données 3D volumiques pour ParaView et logiciels de CAO/SIG 3D.
    """

    def __init__(self, origin: Tuple[float, float, float] = (0.0, 0.0, 0.0), spacing: Tuple[float, float, float] = (20.0, 20.0, 15.0)):
        self.origin = origin
        self.spacing = spacing

    def export_vti(
        self,
        temperature_3d: torch.Tensor,
        u_3d: torch.Tensor,
        v_3d: torch.Tensor,
        w_3d: torch.Tensor,
        fuel_3d: torch.Tensor,
        output_path: str = "reports/3d_cfd/fire_plume_3d.vti"
    ) -> Path:
        """
        Génère un fichier .vti structuré compatible ParaView.
        Format des tenseurs : [D (Z), H (Y), W (X)]
        """
        D, H, W = temperature_3d.shape
        dx, dy, dz = self.spacing
        ox, oy, oz = self.origin

        # Conversion numpy
        T_arr = temperature_3d.detach().cpu().numpy().astype(np.float32)
        u_arr = u_3d.detach().cpu().numpy().astype(np.float32)
        v_arr = v_3d.detach().cpu().numpy().astype(np.float32)
        w_arr = w_3d.detach().cpu().numpy().astype(np.float32)
        fuel_arr = fuel_3d.detach().cpu().numpy().astype(np.float32)

        # Vecteur vitesse combiné [N, 3] en ordre Fortran/VTK (X vite, puis Y, puis Z)
        vel_vector = np.stack([u_arr, v_arr, w_arr], axis=-1)

        # Construction du document XML VTK ImageData
        root = ET.Element("VTKFile", type="ImageData", version="0.1", byte_order="LittleEndian")
        image_data = ET.SubElement(
            root, "ImageData",
            WholeExtent=f"0 {W-1} 0 {H-1} 0 {D-1}",
            Origin=f"{ox} {oy} {oz}",
            Spacing=f"{dx} {dy} {dz}"
        )
        piece = ET.SubElement(image_data, "Piece", Extent=f"0 {W-1} 0 {H-1} 0 {D-1}")
        point_data = ET.SubElement(piece, "PointData", Vectors="Velocity")

        # 1. Champ scalaire Température
        t_data = ET.SubElement(point_data, "DataArray", type="Float32", Name="Temperature_K", format="ascii")
        t_data.text = " ".join(f"{val:.2f}" for val in T_arr.flatten(order="F"))

        # 2. Champ scalaire Densité de Combustible
        f_data = ET.SubElement(point_data, "DataArray", type="Float32", Name="SolidFuel_kg_m3", format="ascii")
        f_data.text = " ".join(f"{val:.3f}" for val in fuel_arr.flatten(order="F"))

        # 3. Champ vectoriel Vitesse 3D (u, v, w)
        v_data = ET.SubElement(point_data, "DataArray", type="Float32", Name="Velocity_3D", NumberOfComponents="3", format="ascii")
        vel_flat = vel_vector.reshape(-1, 3, order="F")
        v_data.text = " ".join(f"{row[0]:.2f} {row[1]:.2f} {row[2]:.2f}" for row in vel_flat)

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(root)
        tree.write(out_file, encoding="utf-8", xml_declaration=True)

        return out_file
