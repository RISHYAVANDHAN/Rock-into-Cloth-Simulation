#ifndef INTEGRATOR_CUH
#define INTEGRATOR_CUH

#include "cloth.cuh"

__global__ void verletUpdatePosition(ClothNode* nodes, float3* old_accel, float dt);

__global__ void verletUpdateVelocity(ClothNode* nodes, float3* old_accel, float dt, float inv_mass);

__global__ void storeAcceleration(ClothNode* nodes, float3* accel_out, float inv_mass);

#endif
