// ==================== energy.cu ====================
#include "energy.cuh"
#include "params.cuh"

__global__ void computeKineticEnergy(ClothNode* nodes, float* kinetic_energy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (idx >= total) return;

    __shared__ float shared_ke[256];
    int tid = threadIdx.x;
    
    shared_ke[tid] = 0.0f;
    
    if (idx < total && !nodes[idx].pinned) {
        float vel_sq = dot(nodes[idx].vel, nodes[idx].vel);
        shared_ke[tid] = 0.5f * node_mass * vel_sq;
    }
    
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_ke[tid] += shared_ke[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(kinetic_energy, shared_ke[0]);
    }
}

__global__ void computePotentialEnergy(ClothNode* nodes, Spring* springs, int numSprings, float* potential_energy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSprings) return;

    __shared__ float shared_pe[256];
    int tid = threadIdx.x;
    
    shared_pe[tid] = 0.0f;
    
    if (idx < numSprings) {
        Spring spring = springs[idx];
        int i = spring.i;
        int j = spring.j;
        float L0 = spring.restLength;
        
        float3 dir = nodes[j].pos - nodes[i].pos;
        float dist = length(dir);
        float strain = dist - L0;
        
        float ks = 0.0f;
        switch (spring.type) {
        case 0: ks = ks_structural; break;
        case 1: ks = ks_shear; break;
        case 2: ks = ks_bend; break;
        }
        // Spring potential energy: U = 0.5 * k * (strain)^2
        shared_pe[tid] = 0.5f * ks * strain * strain;
        
        // Gravitational potential energy (only count once per node)
        if (idx < num_x * num_y) {
            shared_pe[tid] += node_mass * (-gravity.y) * nodes[i].pos.y;
        }
    }
    
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_pe[tid] += shared_pe[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(potential_energy, shared_pe[0]);
    }
}

__global__ void resetEnergyBuffers(float* kinetic_energy, float* potential_energy) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *kinetic_energy = 0.0f;
        *potential_energy = 0.0f;
    }
}
