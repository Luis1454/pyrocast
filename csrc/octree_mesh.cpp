/**
 * csrc/octree_mesh.cpp
 * --------------------
 * Implémentation C++ du maillage spatial Octree adaptatif.
 */

#include "octree_mesh.hpp"
#include <omp.h>
#include <iostream>

namespace firemap {

OctreeGraphMesh FastOctreeBuilder::build(const float* dense_grid, int C, int D, int H, int W) {
    OctreeGraphMesh mesh;
    std::vector<Node>& nodes = mesh.nodes;

    // Indexation par canal dans la grille dense C x D x H x W
    auto idx = [=](int c, int z, int y, int x) -> size_t {
        return ((static_cast<size_t>(c) * D + z) * H + y) * W + x;
    };

    // 1. Calcul du gradient thermique 3D
    std::vector<float> grad_mag(D * H * W, 0.0f);

    #pragma omp parallel for collapse(3) schedule(static)
    for (int z = 0; z < D; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float gz = 0.0f;
                if (D > 1) {
                    int z_prev = std::max(0, z - 1);
                    int z_next = std::min(D - 1, z + 1);
                    gz = (dense_grid[idx(0, z_next, y, x)] - dense_grid[idx(0, z_prev, y, x)]) * 0.5f;
                }

                int y_prev = std::max(0, y - 1);
                int y_next = std::min(H - 1, y + 1);
                float gy = (dense_grid[idx(0, z, y_next, x)] - dense_grid[idx(0, z, y_prev, x)]) * 0.5f;

                int x_prev = std::max(0, x - 1);
                int x_next = std::min(W - 1, x + 1);
                float gx = (dense_grid[idx(0, z, y, x_next)] - dense_grid[idx(0, z, y, x_prev)]) * 0.5f;

                float temp_val = dense_grid[idx(0, z, y, x)];
                float g_norm = std::sqrt(gx * gx + gy * gy + gz * gz);
                
                // Critère de raffinement
                grad_mag[(z * H + y) * W + x] = g_norm + (temp_val > 1.0f ? 1.5f : 0.0f);
            }
        }
    }

    // 2. Échantillonnage hiérarchique des nœuds de l'Octree
    for (int depth = min_depth_; depth <= max_depth_; ++depth) {
        int stride = 1 << (max_depth_ - depth);

        for (int z = 0; z < D; z += stride) {
            for (int y = 0; y < H; y += stride) {
                for (int x = 0; x < W; x += stride) {
                    float crit = grad_mag[(z * H + y) * W + x];
                    bool select = false;

                    if (depth == max_depth_) {
                        select = (crit >= grad_threshold_);
                    } else if (depth == min_depth_) {
                        select = (crit < grad_threshold_ * 0.5f);
                    } else {
                        float lower = grad_threshold_ / (1 << (max_depth_ - depth));
                        select = (crit >= lower && crit < grad_threshold_);
                    }

                    if (select) {
                        Node node;
                        node.x = (static_cast<float>(x) / std::max(W - 1, 1)) * 2.0f - 1.0f;
                        node.y = (static_cast<float>(y) / std::max(H - 1, 1)) * 2.0f - 1.0f;
                        node.z = (static_cast<float>(z) / std::max(D - 1, 1)) * 2.0f - 1.0f;
                        node.level = depth;
                        node.cell_id = static_cast<int32_t>(morton_encode_3d(x, y, z));

                        for (int c = 0; c < std::min(C, 7); ++c) {
                            node.features[c] = dense_grid[idx(c, z, y, x)];
                        }

                        nodes.push_back(node);
                    }
                }
            }
        }
    }

    // 3. Construction des arêtes locales Near-field O(N log N)
    const size_t N = nodes.size();
    float r_sq = r_near_ * r_near_;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            float dx = nodes[j].x - nodes[i].x;
            float dy = nodes[j].y - nodes[i].y;
            float dz = nodes[j].z - nodes[i].z;
            float d2 = dx * dx + dy * dy + dz * dz;

            if (d2 <= r_sq && d2 > 1e-8f) {
                float dist = std::sqrt(d2);
                Edge e1{static_cast<int32_t>(i), static_cast<int32_t>(j), dist, {dx, dy, dz}};
                Edge e2{static_cast<int32_t>(j), static_cast<int32_t>(i), dist, {-dx, -dy, -dz}};
                mesh.near_edges.push_back(e1);
                mesh.near_edges.push_back(e2);
            }
        }
    }

    return mesh;
}

} // namespace firemap
