// ==================== cloth.cu ====================
#include "cloth.cuh"
#include <math.h>

__host__ void initializeClothGrid(ClothNode* d_nodes, int num_x, int num_y, float clothWidth, float clothHeight) {
    ClothNode* h_nodes = new ClothNode[num_x * num_y];

    const float halfWidth = clothWidth / 2.0f;
    const float halfHeight = clothHeight / 2.0f;

    for (int y = 0; y < num_y; ++y) {
        for (int x = 0; x < num_x; ++x) {
            int i = y * num_x + x;

            float rand_z = 0.001f * ((rand() % 100) / 100.0f - 0.5f);  // Z-perturbation
            h_nodes[i].index = i;
            h_nodes[i].pos = make_float3(
                x * (clothWidth / (num_x - 1)) - halfWidth,
                y * (clothHeight / (num_y - 1)) - halfHeight,
                rand_z
            );
            h_nodes[i].vel = make_float3(0.0f, 0.0f, 0.0f);
            h_nodes[i].force = make_float3(0.0f, 0.0f, 0.0f);

            h_nodes[i].pinned =
                (x == 0 && y == 0) ||
                (x == num_x - 1 && y == 0) ||
                (x == 0 && y == num_y - 1) ||
                (x == num_x - 1 && y == num_y - 1);
        }
    }

    cudaMemcpy(d_nodes, h_nodes, num_x * num_y * sizeof(ClothNode), cudaMemcpyHostToDevice);
    delete[] h_nodes;
}

__global__ void applyGravity(ClothNode* nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (i < total && !nodes[i].pinned) {
        nodes[i].force += node_mass * gravity;
    }
}

__global__ void applyViscousDamping(ClothNode* nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (i < total && !nodes[i].pinned) {
        nodes[i].force += -kd * nodes[i].vel;
    }
}

__global__ void resetClothForces(ClothNode* nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (i < total) {
        nodes[i].force = make_float3(0.f, 0.f, 0.f);
    }
}

__global__ void applyPinningConstraints(ClothNode* nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (i < total && nodes[i].pinned) {
        nodes[i].vel = make_float3(0.f, 0.f, 0.f);
        nodes[i].force = make_float3(0.f, 0.f, 0.f);
    }
}

__global__ void applySpringForces(ClothNode* nodes, Spring* springs, int numSprings) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSprings) return;

    Spring spring = springs[idx];
    int i = spring.i;
    int j = spring.j;
    float L0 = spring.restLength;

    float3 xi = nodes[i].pos;
    float3 xj = nodes[j].pos;
    float3 vi = nodes[i].vel;
    float3 vj = nodes[j].vel;

    float3 dir = xj - xi;
    float dist = length(dir);
    if (dist < 1e-6f) return;

    float3 n = dir / dist;

    float ks_local = 0.0f;
    switch (spring.type) {
        case 0: ks_local = ks_structural; break;
        case 1: ks_local = ks_shear; break;
        case 2: ks_local = ks_bend; break;
    }

    float3 f_spring = -ks_local * ((dist - L0) / L0) * n;

    float3 v_rel = vj - vi;
    float v_along_spring = dot(v_rel, n);
    float3 f_damp = -kd * v_along_spring * n;

    float3 f_total = f_spring + f_damp;

    if (!nodes[i].pinned) {
        atomicAdd(&nodes[i].force.x, -f_total.x);
        atomicAdd(&nodes[i].force.y, -f_total.y);
        atomicAdd(&nodes[i].force.z, -f_total.z);
    }
    if (!nodes[j].pinned) {
        atomicAdd(&nodes[j].force.x, f_total.x);
        atomicAdd(&nodes[j].force.y, f_total.y);
        atomicAdd(&nodes[j].force.z, f_total.z);
    }
}
