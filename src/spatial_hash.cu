#include "spatial_hash.cuh"

__device__ __host__ uint calcHash(int3 cell, int3 gridSize) {
    // Clamp cell coordinates to grid boundaries
    cell.x = max(0, min(cell.x, gridSize.x - 1));
    cell.y = max(0, min(cell.y, gridSize.y - 1));
    cell.z = max(0, min(cell.z, gridSize.z - 1));
    
    // Compute linear hash index using row-major order
    return (cell.z * gridSize.y + cell.y) * gridSize.x + cell.x;
}

__device__ __host__ int3 computeCellIndex(float3 p, float3 domainMin, float cellWidth) {
    return make_int3(
        floorf((p.x - domainMin.x) / cellWidth),
        floorf((p.y - domainMin.y) / cellWidth),
        floorf((p.z - domainMin.z) / cellWidth)
    );
}

__global__ void resetCellHeads(int* cellHead, int numCells) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numCells) {
        cellHead[i] = -1;  // -1 indicates empty cell
    }
}

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
    
    // Compute grid cell for this position
    int3 cell = computeCellIndex(pos, gridMin, cellSize);

    // Check if cell is within grid bounds
    if (cell.x < 0 || cell.y < 0 || cell.z < 0 ||
        cell.x >= gridSize.x || cell.y >= gridSize.y || cell.z >= gridSize.z) {
        nodeNext[i] = -1;  // Mark as invalid
        return;
    }

    // Compute hash for this cell
    uint hash = calcHash(cell, gridSize);

    // Atomically insert node at head of linked list
    // This prepends the current node to the cell's linked list
    nodeNext[i] = atomicExch(&cellHead[hash], i);
}

__global__ void insertMarblesToGrid(
    Marble* marbles, 
    int* cellHead, 
    int* marbleNodeNext,
    int numMarbles, 
    int3 gridSize, 
    float3 gridMin, 
    float cellSize
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numMarbles) return;

    float3 pos = marbles[i].pos;
    
    // Compute grid cell for this marble
    int3 gridPos = computeCellIndex(pos, gridMin, cellSize);

    // Check bounds
    if (gridPos.x < 0 || gridPos.y < 0 || gridPos.z < 0 ||
        gridPos.x >= gridSize.x || gridPos.y >= gridSize.y || gridPos.z >= gridSize.z) {
        marbleNodeNext[i] = -1;  // Mark as invalid
        return;
    }

    // Compute hash and insert into linked list
    uint hash = calcHash(gridPos, gridSize);
    marbleNodeNext[i] = atomicExch(&cellHead[hash], i);
}