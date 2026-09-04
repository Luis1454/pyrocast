"""
src/export/tensorrt_export.py
-----------------------------
Pipeline d'optimisation et d'exportation pour l'inférence en production :
- Export TorchScript (.pt) pour intégration directe dans les binaires C++ LibTorch
- Export ONNX (.onnx)
- Compilation / Script de construction TensorRT pour exécution GPU sous la milliseconde
"""

from pathlib import Path
import torch
import torch.nn as nn
from ..models.mgnn_fmm import NeuralFMM_MGNN
from ..mesh.octree_graph import GraphMesh, AdaptiveOctreeGraphBuilder


class ExportableNeuralFMMWrapper(nn.Module):
    """
    Wrapper compatible avec le traçage ONNX et TorchScript recevant des tenseurs explicites.
    """

    def __init__(self, model: NeuralFMM_MGNN):
        super().__init__()
        self.model = model

    def forward(
        self,
        node_features: torch.Tensor,
        node_coords: torch.Tensor,
        edge_index_near: torch.Tensor,
        edge_index_far: torch.Tensor,
        wind_forcing_3d: torch.Tensor,
    ) -> torch.Tensor:
        mesh = GraphMesh(
            node_features=node_features,
            node_coords=node_coords,
            node_levels=torch.zeros(node_features.shape[0], dtype=torch.long, device=node_features.device),
            edge_index_near=edge_index_near,
            edge_index_far=edge_index_far,
        )
        return self.model(mesh, wind_forcing_3d)


def export_model_to_torchscript_and_onnx(
    output_dir: str = "reports/export",
    device: str = "cpu"
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device)

    print(f"[*] Initialisation de l'exportation du modèle sur : {dev}")

    # 1. Modèle et données fictives
    raw_model = NeuralFMM_MGNN(in_channels=7, hidden_dim=64, out_channels=7, num_layers=3, forcing_dim=3).to(dev).eval()
    wrapper = ExportableNeuralFMMWrapper(raw_model).to(dev).eval()

    # Données d'exemple
    N_nodes = 256
    E_near = 1024
    E_far = 512

    dummy_features = torch.randn(N_nodes, 7, device=dev)
    dummy_coords = torch.randn(N_nodes, 3, device=dev)
    dummy_edge_near = torch.randint(0, N_nodes, (2, E_near), device=dev)
    dummy_edge_far = torch.randint(0, N_nodes, (2, E_far), device=dev)
    dummy_wind = torch.tensor([1.5, -0.5, 0.2], device=dev)

    # 2. Export TorchScript
    ts_path = out_dir / "neural_fmm_mgnn.pt"
    print(f"[*] Traçage TorchScript...")
    traced_model = torch.jit.trace(
        wrapper,
        (dummy_features, dummy_coords, dummy_edge_near, dummy_edge_far, dummy_wind)
    )
    traced_model.save(str(ts_path))
    print(f"[OK] Modèle TorchScript exporté : {ts_path}")

    # 3. Export ONNX (si onnx / onnxscript installés)
    onnx_path = out_dir / "neural_fmm_mgnn.onnx"
    try:
        print(f"[*] Exportation ONNX...")
        torch.onnx.export(
            wrapper,
            (dummy_features, dummy_coords, dummy_edge_near, dummy_edge_far, dummy_wind),
            str(onnx_path),
            input_names=["node_features", "node_coords", "edge_near", "edge_far", "wind_3d"],
            output_names=["next_node_state"],
            dynamic_axes={
                "node_features": {0: "num_nodes"},
                "node_coords": {0: "num_nodes"},
                "edge_near": {1: "num_edges_near"},
                "edge_far": {1: "num_edges_far"},
                "next_node_state": {0: "num_nodes"}
            },
            opset_version=17
        )
        print(f"[OK] Modèle ONNX exporté : {onnx_path}")
    except (ImportError, Exception) as e:
        print(f"[!] Export ONNX non disponible ({e}), utilisation prioritaire de TorchScript (.pt) pour C++ / TensorRT.")

    # 4. Script de génération TensorRT
    trt_builder_script = out_dir / "build_tensorrt_engine.sh"
    with open(trt_builder_script, "w") as f:
        f.write(f"""#!/bin/bash
# Construction du moteur TensorRT optimisé FP16 sur GPU cible
trtexec --onnx={onnx_path.name} \\
        --saveEngine=neural_fmm_mgnn_fp16.engine \\
        --fp16 \\
        --minShapes=node_features:32x7,node_coords:32x3,edge_near:2x64,edge_far:2x32,wind_3d:3 \\
        --optShapes=node_features:2048x7,node_coords:2048x3,edge_near:2x8192,edge_far:2x2048,wind_3d:3 \\
        --maxShapes=node_features:16384x7,node_coords:16384x3,edge_near:2x65536,edge_far:2x16384,wind_3d:3
""")
    print(f"[OK] Script de compilation TensorRT généré : {trt_builder_script}")


if __name__ == "__main__":
    export_model_to_torchscript_and_onnx()
