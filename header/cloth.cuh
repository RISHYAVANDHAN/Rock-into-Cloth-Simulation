#ifndef CLOTH_CUH
#define CLOTH_CUH

#include <cuda_runtime.h>
#include "float3_utils.cuh"
#include "params.cuh"

struct ClothNode {
    int index;     // Index of the cloth node
    float3 pos;    // Position in world space
    float3 vel;    // Velocity
    float3 force;  // Accumulated force
    bool pinned;   // Is this node fixed in place?
};

struct Spring {
    int i;              // Index of node i
    int j;              // Index of node j
    float restLength;   // Initial length of the spring
};

// Force computation kernels
__global__ void applySpringForces(ClothNode* nodes, Spring* springs, int numSprings);

__global__ void applyGravity(ClothNode* nodes);

__global__ void applyViscousDamping(ClothNode* nodes);

__global__ void resetClothForces(ClothNode* nodes);

__global__ void applyPinningConstraints(ClothNode* nodes);

__host__ void initializeClothGrid(ClothNode* d_nodes, int num_x, int num_y, float clothWidth, float clothHeight);

#endif // CLOTH_CUH
