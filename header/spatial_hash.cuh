#ifndef SPATIAL_HASH_CUH
#define SPATIAL_HASH_CUH

#include <cuda_runtime.h>
#include "float3_utils.cuh"
#include "marble.cuh"

#define MAX_PARTICLES_PER_CELL 64

// Global variables (extern declarations)
extern float cell_size;
extern int3 grid_res;
extern int *cellCounts;
extern int *cellContents;
extern int *marbleCellIdx;

// Device function to calculate hash from grid position
__device__ __host__ uint calcHash(int3 cell, int3 gridSize);

// Utility function for cell index computation
__device__ __host__ int3 computeCellIndex(float3 p, float3 domainMin, float cellWidth);

// Kernel to build spatial linked list for generic positions
__global__ void buildSpatialLinkedList(
    const float3* positions,
    int* cellHead,
    int* nodeNext,
    int numNodes,
    int3 gridSize,
    float3 gridMin,
    float cellSize
);

// Kernel to insert marbles into grid using linked list
__global__ void insertMarblesToGrid(
    Marble* marbles, 
    int* cellHead, 
    int* marbleNext,
    int numMarbles, 
    int3 gridSize, 
    float3 gridMin, 
    float cellSize
);

// Kernel to reset cell heads (initialize linked list)
__global__ void resetCellHeads(int* cellHead, int numCells);

#endif