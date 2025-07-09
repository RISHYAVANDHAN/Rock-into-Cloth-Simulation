#ifndef CLOTH_CUH
#define CLOTH_CUH

#include <cuda_runtime.h>
#include "float3_utils.cuh"
#include "params.cuh"

struct ClothNode {
    int index;
    float3 pos;
    float3 vel;
    float3 force;
    bool pinned;
};

enum SpringType {
    STRUCTURAL = 0,
    SHEAR = 1,
    BENDING = 2
};

struct Spring {
    int i;
    int j;
    float restLength;
    int type;
};

__global__ void applySpringForces(ClothNode* nodes, Spring* springs, int numSprings);

__global__ void applyGravity(ClothNode* nodes);

__global__ void applyViscousDamping(ClothNode* nodes);

__global__ void resetClothForces(ClothNode* nodes);

__global__ void applyPinningConstraints(ClothNode* nodes);

__host__ void initializeClothGrid(ClothNode* d_nodes, int num_x, int num_y, float clothWidth, float clothHeight);

#endif
