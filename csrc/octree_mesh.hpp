/**
 * csrc/octree_mesh.hpp
 * --------------------
 * Moteur C++ haute performance pour la construction d'Octree adaptatif et de graphe FMM.
 * Utilise l'encodage par codes de Morton (Z-order curve) et le parallélisme OpenMP.
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <memory>

namespace firemap {

struct Node {
    float x, y, z;
    int32_t level;
    int32_t cell_id;
    float features[7]; // [Theta, P, U, V, W, FMC, FireFlux]
};

struct Edge {
    int32_t src;
    int32_t dst;
    float dist;
    float dr[3];
};

struct OctreeGraphMesh {
    std::vector<Node> nodes;
    std::vector<Edge> near_edges;
    std::vector<Edge> far_edges;
};

class FastOctreeBuilder {
public:
    FastOctreeBuilder(int max_depth = 4, int min_depth = 1, float grad_threshold = 0.25f, float r_near = 0.12f, int k_far = 6)
        : max_depth_(max_depth), min_depth_(min_depth), grad_threshold_(grad_threshold), r_near_(r_near), k_far_(k_far) {}

    // Encodage Morton 3D (Z-Order Curve) sur 64 bits pour localisation spatiale ultra-rapide
    static inline uint64_t morton_encode_3d(uint32_t x, uint32_t y, uint32_t z) {
        auto spread_bits = [](uint32_t v) -> uint64_t {
            uint64_t x = v & 0x1fffff;
            x = (x | (x << 32)) & 0x1f00000000ffff;
            x = (x | (x << 16)) & 0x1f0000ff0000ff;
            x = (x | (x << 8))  & 0x100f00f00f00f00f;
            x = (x | (x << 4))  & 0x10c30c30c30c30c3;
            x = (x | (x << 2))  & 0x1249249249249249;
            return x;
        };
        return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2);
    }

    // Construction du maillage et des arêtes
    OctreeGraphMesh build(const float* dense_grid, int C, int D, int H, int W);

private:
    int max_depth_;
    int min_depth_;
    float grad_threshold_;
    float r_near_;
    int k_far_;
};

} // namespace firemap
