/**
 * csrc/fast_multipole.cu
 * ----------------------
 * Kernel CUDA pour l'accélération massive des interactions multipôles (FMM)
 * et du transport radiatif Stefan-Boltzmann non-linéaire sur GPU.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace firemap {
namespace cuda {

__global__ void FMM_MultipoleRadiationKernel(
    const float* __restrict__ node_coords,     // [N, 3]
    const float* __restrict__ node_features,   // [N, C] (Canal 0 = Température)
    const int32_t* __restrict__ edge_index_far,// [2, E_far]
    float* __restrict__ output_radiation,      // [N, C]
    int num_edges,
    float sigma_sb,
    float emissivity
) {
    int edge_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_id >= num_edges) return;

    int src = edge_index_far[edge_id];
    int dst = edge_index_far[num_edges + edge_id];

    // Coordonnées 3D
    float src_x = node_coords[src * 3 + 0];
    float src_y = node_coords[src * 3 + 1];
    float src_z = node_coords[src * 3 + 2];

    float dst_x = node_coords[dst * 3 + 0];
    float dst_y = node_coords[dst * 3 + 1];
    float dst_z = node_coords[dst * 3 + 2];

    float dx = dst_x - src_x;
    float dy = dst_y - src_y;
    float dz = dst_z - src_z;
    float dist_sq = dx * dx + dy * dy + dz * dz + 1e-4f;

    // Température du multipôle source T_src
    float t_src = node_features[src * 7 + 0]; // Normalisé

    // Atténuation géométrique en 1 / (4 * pi * r^2)
    float geometric_factor = 1.0f / (12.56637f * dist_sq);
    float q_rad = emissivity * sigma_sb * (t_src * t_src * t_src * t_src) * geometric_factor;

    // Accumulation atomique sur le nœud récepteur
    atomicAdd(&output_radiation[dst * 7 + 0], q_rad);
}

void launch_fmm_cuda(
    const float* node_coords,
    const float* node_features,
    const int32_t* edge_index_far,
    float* output_radiation,
    int num_edges,
    cudaStream_t stream = 0
) {
    int threads_per_block = 256;
    int num_blocks = (num_edges + threads_per_block - 1) / threads_per_block;

    FMM_MultipoleRadiationKernel<<<num_blocks, threads_per_block, 0, stream>>>(
        node_coords, node_features, edge_index_far, output_radiation, num_edges, 5.67e-8f, 0.95f
    );
}

} // namespace cuda
} // namespace firemap
