#include "../header/applied_force.cuh"
#include "../header/float3_utils.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

float3* d_externalForce = nullptr;

void allocateExternalForceBuffer() {
    int total = 10000;
    cudaMalloc(&d_externalForce, total * sizeof(float3));
    cudaMemset(d_externalForce, 0, total * sizeof(float3));
}

void freeExternalForceBuffer() {
    if (d_externalForce) {
        cudaFree(d_externalForce);
        d_externalForce = nullptr;
    }
}

__global__ void kernelInjectForces(ClothNode* nodes, const float3* externalForces, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        nodes[i].force += externalForces[i];
    }
}

__global__ void kernelResetExternalForces(float3* externalForces, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        externalForces[i] = make_float3(0.0f, 0.0f, 0.0f);
    }
}

void applyPrescribedForce(int step, float3* d_externalForces, int Nx, int Ny, const std::string& mode,
                          float strength, int radius, int start_step, int end_step) {
    if (step < start_step || step > end_step) return;

    std::vector<float3> h_force(Nx * Ny, make_float3(0.0f, 0.0f, 0.0f));
    int cx = Nx / 2;
    int cy = Ny / 2;

    float phase = 4.0f * M_PI * (step - start_step) / (end_step - start_step);
    float oscillation = sin(phase);
    float directional_strength = strength;

    for (int j = 0; j < Ny; ++j) {
        for (int i = 0; i < Nx; ++i) {
            int id = j * Nx + i;
            float3 force = make_float3(0.0f, 0.0f, 0.0f);

            if (mode == "point") {
                if (i == cx && j == cy)
                    force = make_float3(0.0f, directional_strength, 0.0f);
            } else if (mode == "patch") {
                if (abs(i - cx) <= radius && abs(j - cy) <= radius)
                    force = make_float3(0.0f, directional_strength, 0.0f);
            } else if (mode == "line") {
                if (j == cy)
                    force = make_float3(0.0f, directional_strength, 0.0f);
            } else if (mode == "gaussian") {
                float dx = float(i - cx);
                float dy = float(j - cy);
                float dist2 = dx * dx + dy * dy;
                float sigma2 = float(radius * radius);
                float factor = exp(-dist2 / (2.0f * sigma2));
                force = make_float3(0.0f, directional_strength * factor, 0.0f);
            }

            h_force[id] = force;
        }
    }

    cudaMemcpy(d_externalForces, h_force.data(), Nx * Ny * sizeof(float3), cudaMemcpyHostToDevice);
}
