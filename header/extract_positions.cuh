#ifndef EXTRACT_POSITIONS_CUH
#define EXTRACT_POSITIONS_CUH

#include <cuda_runtime.h>
#include "cloth.cuh"

__global__ void extractPositions(const ClothNode* nodes, float3* positions, int total);

#endif