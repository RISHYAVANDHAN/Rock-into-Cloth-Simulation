#ifndef APPLIED_FORCE_CUH
#define APPLIED_FORCE_CUH

#include <cuda_runtime.h>
#include <string>
#include "cloth.cuh"  // Added to resolve ClothNode dependency

extern float3* d_externalForce;

void allocateExternalForceBuffer();
void freeExternalForceBuffer();

__device__ __host__ inline int idx(int x, int y, int nx) {
    return y * nx + x;
}

void applyPrescribedForce(int step,
                          float3* d_externalForces,
                          int Nx, int Ny,
                          const std::string& mode,
                          float strength,
                          int radius,
                          int start_step,
                          int end_step);

// Changed first parameter to ClothNode* to match implementation
__global__ void kernelInjectForces(ClothNode* nodes, const float3* externalForces, int total);

// Changed parameter name to be consistent
__global__ void kernelResetExternalForces(float3* externalForces, int total);

#endif