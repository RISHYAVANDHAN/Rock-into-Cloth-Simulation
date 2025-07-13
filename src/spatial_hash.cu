#include "spatial_hash.cuh"

__global__ void buildSpatialLinkedList(
    const float3* positions,
    int* cellHead,
    int* nodeNext,
    int numNodes,
    int3 gridSize,
    float3 gridMin,
    float cellSize
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes) return;

    float3 pos = positions[i];
    int3 cell = make_int3(
        floorf((pos.x - gridMin.x) / cellSize),
        floorf((pos.y - gridMin.y) / cellSize),
        floorf((pos.z - gridMin.z) / cellSize)
    );

    if (cell.x < 0 || cell.y < 0 || cell.z < 0 ||
        cell.x >= gridSize.x || cell.y >= gridSize.y || cell.z >= gridSize.z)
        return;

    int hash = computeHash(cell, gridSize);

    // Insert node into linked list (atomic prepend)
    nodeNext[i] = atomicExch(&cellHead[hash], i);
}

__global__ void insertMarblesToGrid(
    Marble* marbles, int* cellHead, int* marbleNext,
    int numMarbles, int3 gridSize, float3 gridMin, float cellSize
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numMarbles) return;

    float3 pos = marbles[i].pos;
    int3 gridPos = make_int3(
        floorf((pos.x - gridMin.x) / cellSize),
        floorf((pos.y - gridMin.y) / cellSize),
        floorf((pos.z - gridMin.z) / cellSize)
    );

    if (gridPos.x < 0 || gridPos.y < 0 || gridPos.z < 0 ||
        gridPos.x >= gridSize.x || gridPos.y >= gridSize.y || gridPos.z >= gridSize.z) return;

    int hash = computeHash(gridPos, gridSize);
    marbleNext[i] = atomicExch(&cellHead[hash], i);
}