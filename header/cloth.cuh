// ==================== cloth.cuh ====================
#ifndef CLOTH_CUH
#define CLOTH_CUH

#pragma once

#include <cuda_runtime.h>
#include "float3_utils.cuh"
#include "params.cuh"

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

// Function declarations for cloth simulation
__host__ void initializeClothGrid(ClothNode* d_nodes, int num_x, int num_y, float clothWidth, float clothHeight);

// Force application kernels
__global__ void applyGravity(ClothNode* nodes);

__global__ void applyViscousDamping(ClothNode* nodes);

__global__ void resetClothForces(ClothNode* nodes);

__global__ void applyPinningConstraints(ClothNode* nodes);

__global__ void applySpringForces(ClothNode* nodes, Spring* springs, int numSprings);

// Collision detection kernels
__global__ void applyClothSelfContact(ClothNode* nodes, const int* cellHead, const int* nodeNext, int3 gridSize, float3 gridMin, float cellSize);

__global__ void applyMarbleClothContact(Marble* marbles, int numMarbles, ClothNode* nodes, const int* cellHead, const int* nodeNext, int3 gridSize, float3 gridMin, float cellSize);

__global__ void applyMarbleMarbleContact(Marble* marbles, int numMarbles, const int* cellHead, const int* marbleNext, int3 gridSize, float3 gridMin, float cellSize);

__global__ void limitClothNodeVelocities(ClothNode* nodes, int numNodes, float maxVel);

__global__ void resetClothForces(ClothNode* nodes, int numNodes);

#endif