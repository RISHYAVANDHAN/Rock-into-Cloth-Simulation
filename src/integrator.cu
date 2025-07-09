#include "integrator.cuh"
#include "params.cuh"

__global__ void verletUpdatePosition(ClothNode* nodes, float3* old_accel, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (i < total && !nodes[i].pinned) {
        float3 v = nodes[i].vel;
        float3 a = old_accel[i];
        nodes[i].pos += v * dt + 0.5f * a * dt * dt;
    }
}

__global__ void verletUpdateVelocity(ClothNode* nodes, float3* old_accel, float dt, float inv_mass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (i < total && !nodes[i].pinned) {
        float3 a_old = old_accel[i];
        float3 a_new = nodes[i].force * inv_mass;
        nodes[i].vel += 0.5f * (a_old + a_new) * dt;
    }
}

__global__ void storeAcceleration(ClothNode* nodes, float3* accel_out, float inv_mass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (i < total) {
        accel_out[i] = nodes[i].force * inv_mass;
    }
}

// DEM-based marble–cloth contact (1 sphere version)
__global__ void applyMarbleCollision(ClothNode* nodes, float3 center, float radius, float kn, float kd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (i >= total || nodes[i].pinned) return;

    float3 dir = nodes[i].pos - center;
    float dist = length(dir);
    if (dist < radius) {
        float3 normal = normalize(dir);
        float3 vrel = nodes[i].vel;
        float vrel_n = dot(vrel, normal);
        float3 f_contact = kn * (radius - dist) * normal - kd * vrel_n * normal;
        nodes[i].force += f_contact;
    }
}
