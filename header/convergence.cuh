
#ifndef CONVERGENCE_CUH
#define CONVERGENCE_CUH

#include <cuda_runtime.h>
#include "cloth.cuh"
#include "params.cuh"

struct ConvergenceData {
    float total_velocity;
    float total_force;
    float max_velocity;
    float max_force;
    float avg_velocity;
    float avg_force;
};

__global__ void computeConvergenceMetrics(ClothNode* nodes, ConvergenceData* conv_data);

__global__ void resetConvergenceData(ConvergenceData* conv_data);

bool checkConvergence(ConvergenceData* d_conv_data, int total_nodes, float vel_threshold, float force_threshold);

#endif