// ==================== cloth.cuh ====================
#ifndef CLOTH_CUH
#define CLOTH_CUH

#include <cuda_runtime.h>
#include "float3_utils.cuh"
#include "params.cuh"
#include "marble.cuh"

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

// Add to existing SpringType enum
enum ContactType {
    CLOTH_MARBLE = 0,
    // Add other contact types if needed
};

__global__ void computeMarbleClothInteraction(ClothNode* nodes, Marble* marbles, int numMarbles, float dt);

__global__ void applySpringForces(ClothNode* nodes, Spring* springs, int numSprings);

__global__ void applyGravity(ClothNode* nodes);

__global__ void applyViscousDamping(ClothNode* nodes);

__global__ void resetClothForces(ClothNode* nodes);

__global__ void applyPinningConstraints(ClothNode* nodes);

__host__ void initializeClothGrid(ClothNode* d_nodes, int num_x, int num_y, float clothWidth, float clothHeight);

__global__ void applyClothSelfContact(ClothNode* nodes, const int* cellHead, const int* nodeNext, int3 gridSize, float3 gridMin, float cellSize);

__global__ void computeMarbleClothInteraction(ClothNode* nodes, Marble* marbles, int numMarbles, float dt);

__global__ void applyMarbleMarbleContact(Marble* marbles, int numMarbles, const int* cellHead, const int* marbleNext, int3 gridSize, float3 gridMin, float cellSize);

__global__ void computeMarbleTriangleCollision(ClothNode* nodes,Marble* marbles,int numMarbles);

__device__ void checkSphereTriangleCollision(Marble* marble,ClothNode* v0,ClothNode* v1,ClothNode* v2);


#endif