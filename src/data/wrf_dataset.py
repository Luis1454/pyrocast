"""
src/data/wrf_dataset.py
-----------------------
Ingestion et prétraitement haute performance des simulations WRF-SFIRE (.nc).
Gestion du dé-staggering de la grille Arakawa C-grid et extraction des grandeurs thermodynamiques.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class WRFSFireDataset(Dataset):
    """
    Dataset PyTorch pour simulations WRF-SFIRE au format NetCDF4.
    
    Variables extraites et normalisées :
      0: Theta (Température potentielle absolue T + T00 [K])
      1: Pression totale (P + PB) [Pa]
      2: Vent U dé-staggeré [m/s]
      3: Vent V dé-staggeré [m/s]
      4: Vent W dé-staggeré [m/s]
      5: FMC (Fuel Moisture Content / Humidité combustible) [0, 1]
      6: Fire Flux (Flux de chaleur surfacique dégagé par le feu) [W/m^2]
    """

    def __init__(
        self,
        nc_files: List[Union[str, Path]],
        temporal_window: int = 2,
        stats: Optional[Dict[str, Tuple[float, float]]] = None,
        device: torch.device = torch.device("cpu"),
        precision: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.nc_files = [Path(f) for f in nc_files]
        self.temporal_window = temporal_window
        self.device = device
        self.precision = precision
        
        # Moyenne et écart-type physiques pour normalisation Z-score
        self.stats = stats or {
            "temperature": (300.0, 50.0),   # [K]
            "pressure": (101325.0, 5000.0), # [Pa]
            "wind_u": (0.0, 10.0),          # [m/s]
            "wind_v": (0.0, 10.0),          # [m/s]
            "wind_w": (0.0, 2.0),           # [m/s]
            "moisture": (0.10, 0.08),       # [0, 1]
            "fire_flux": (0.0, 50000.0),    # [W/m^2]
        }

        self.samples: List[Tuple[int, int]] = []
        self._build_index()

    def _build_index(self):
        """Indexe les fichiers et pas temporels pour un accès direct O(1)."""
        for file_idx, fpath in enumerate(self.nc_files):
            with xr.open_dataset(fpath, engine="netcdf4") as ds:
                time_steps = ds.dims.get("Time", ds.dims.get("time", 1))
                for t in range(time_steps - self.temporal_window + 1):
                    self.samples.append((file_idx, t))

    @staticmethod
    def _destagger_arakawa_c(
        u: torch.Tensor, v: torch.Tensor, w: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Réaligne les grandeurs flux discrétisées sur les faces vers le centre de la maille.
        
        Maths :
            u_center[..., x] = 0.5 * (u[..., x] + u[..., x+1])
            v_center[..., y] = 0.5 * (v[..., y] + v[..., y+1])
            w_center[..., z] = 0.5 * (w[..., z] + w[..., z+1])
        """
        u_c = 0.5 * (u[..., :, :, :-1] + u[..., :, :, 1:])
        v_c = 0.5 * (v[..., :, :-1, :] + v[..., :, 1:, :])
        w_c = 0.5 * (w[..., :-1, :, :] + w[..., 1:, :, :]) if w is not None else None
        return u_c, v_c, w_c

    def _normalize(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        mean, std = self.stats[key]
        return (tensor - mean) / (std + 1e-7)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        file_idx, t_start = self.samples[idx]
        nc_path = self.nc_files[file_idx]

        with xr.open_dataset(nc_path, engine="netcdf4") as ds:
            t_slice = slice(t_start, t_start + self.temporal_window)
            
            base_t00 = float(ds.attrs.get("T00", 300.0))
            theta = torch.from_numpy(ds["T"].isel(Time=t_slice).values).to(self.precision) + base_t00
            
            if "PB" in ds and "P" in ds:
                p_tot = torch.from_numpy((ds["P"].isel(Time=t_slice) + ds["PB"].isel(Time=t_slice)).values).to(self.precision)
            else:
                p_tot = torch.zeros_like(theta)

            u_raw = torch.from_numpy(ds["U"].isel(Time=t_slice).values).to(self.precision)
            v_raw = torch.from_numpy(ds["V"].isel(Time=t_slice).values).to(self.precision)
            w_raw = torch.from_numpy(ds["W"].isel(Time=t_slice).values).to(self.precision) if "W" in ds else None

            u_c, v_c, w_c = self._destagger_arakawa_c(u_raw, v_raw, w_raw)

            fmc = torch.from_numpy(
                ds["FMC_G"].isel(Time=t_slice).values if "FMC_G" in ds 
                else np.full_like(theta.cpu().numpy(), 0.1)
            ).to(self.precision)

            fire_flux = torch.from_numpy(
                ds["GRNHFX"].isel(Time=t_slice).values if "GRNHFX" in ds 
                else np.zeros_like(theta.cpu().numpy())
            ).to(self.precision)

        theta_norm = self._normalize(theta, "temperature")
        p_norm = self._normalize(p_tot, "pressure")
        u_norm = self._normalize(u_c, "wind_u")
        v_norm = self._normalize(v_c, "wind_v")
        w_norm = self._normalize(w_c, "wind_w") if w_c is not None else torch.zeros_like(u_norm)
        fmc_norm = self._normalize(fmc, "moisture")
        flux_norm = self._normalize(fire_flux, "fire_flux")

        # Tensor shape : [T_window, Channels, (Z), Y, X]
        state = torch.stack([
            theta_norm, p_norm, u_norm, v_norm, w_norm, fmc_norm, flux_norm
        ], dim=1).to(self.device)

        return {
            "x_input": state[0],      # État t_0
            "y_target": state[1],     # État t_1 (Vérité terrain)
            "wind_forcing": torch.stack([u_norm[1], v_norm[1]], dim=0) # Forçage météo t_1
        }
