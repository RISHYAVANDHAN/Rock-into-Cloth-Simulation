#ifndef SPATIAL_HASH_CUH
#define SPATIAL_HASH_CUH

#include <cuda_runtime.h>
#include "float3_utils.cuh"
#include "marble.cuh"

__global__ void buildSpatialLinkedList(const float3* positions, int* cellHead, int* nodeNext, int numNodes, int3 gridSize, float3 gridMin, float cellSize);

__device__ __host__ inline int computeHash(int3 cell, int3 gridSize) {
    return cell.z * gridSize.y * gridSize.x + cell.y * gridSize.x + cell.x;
}

__global__ void insertMarblesToGrid(Marble* marbles, int* cellHead, int* nodeNext, int numMarbles, int3 gridSize, float3 gridMin, float cellSize);


#endif