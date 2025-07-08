// ==================== convergence.cu ====================
#include "convergence.cuh"
#include <cfloat>

__global__ void computeConvergenceMetrics(ClothNode* nodes, ConvergenceData* conv_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (idx >= total) return;

    // Initialize shared memory for reduction
    __shared__ float shared_vel_sum[256];
    __shared__ float shared_force_sum[256];
    __shared__ float shared_max_vel[256];
    __shared__ float shared_max_force[256];

    int tid = threadIdx.x;
    
    // Initialize shared memory
    shared_vel_sum[tid] = 0.0f;
    shared_force_sum[tid] = 0.0f;
    shared_max_vel[tid] = 0.0f;
    shared_max_force[tid] = 0.0f;

    if (idx < total && !nodes[idx].pinned) {
        float vel_mag = length(nodes[idx].vel);
        float force_mag = length(nodes[idx].force);
        
        shared_vel_sum[tid] = vel_mag;
        shared_force_sum[tid] = force_mag;
        shared_max_vel[tid] = vel_mag;
        shared_max_force[tid] = force_mag;
    }

    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_vel_sum[tid] += shared_vel_sum[tid + stride];
            shared_force_sum[tid] += shared_force_sum[tid + stride];
            shared_max_vel[tid] = fmaxf(shared_max_vel[tid], shared_max_vel[tid + stride]);
            shared_max_force[tid] = fmaxf(shared_max_force[tid], shared_max_force[tid + stride]);
        }
        __syncthreads();
    }

    // Write block results to global memory
    if (tid == 0) {
        atomicAdd(&conv_data->total_velocity, shared_vel_sum[0]);
        atomicAdd(&conv_data->total_force, shared_force_sum[0]);
        
        // Use union trick for atomic max on float
        int* max_vel_as_int = (int*)&conv_data->max_velocity;
        int* max_force_as_int = (int*)&conv_data->max_force;
        
        atomicMax(max_vel_as_int, __float_as_int(shared_max_vel[0]));
        atomicMax(max_force_as_int, __float_as_int(shared_max_force[0]));
    }
}

__global__ void resetConvergenceData(ConvergenceData* conv_data) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        conv_data->total_velocity = 0.0f;
        conv_data->total_force = 0.0f;
        conv_data->max_velocity = 0.0f;
        conv_data->max_force = 0.0f;
        conv_data->avg_velocity = 0.0f;
        conv_data->avg_force = 0.0f;
    }
}

bool checkConvergence(ConvergenceData* d_conv_data, int total_nodes, 
                     float vel_threshold, float force_threshold) {
    ConvergenceData h_conv_data;
    cudaMemcpy(&h_conv_data, d_conv_data, sizeof(ConvergenceData), cudaMemcpyDeviceToHost);
    
    // Calculate averages
    int free_nodes = total_nodes - 4; // Subtract pinned corners
    h_conv_data.avg_velocity = h_conv_data.total_velocity / free_nodes;
    h_conv_data.avg_force = h_conv_data.total_force / free_nodes;
    
    // Check convergence criteria
    bool vel_converged = h_conv_data.max_velocity < vel_threshold;
    bool force_converged = h_conv_data.max_force < force_threshold;
    
    return (vel_converged || force_converged);
}
