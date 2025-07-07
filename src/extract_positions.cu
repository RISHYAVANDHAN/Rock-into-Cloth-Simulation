#include "../header/extract_positions.cuh"

__global__ void extractPositions(const ClothNode* nodes, float3* positions, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        positions[i] = nodes[i].pos;
    }
}