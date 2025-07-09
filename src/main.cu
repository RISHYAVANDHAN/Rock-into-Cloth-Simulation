// ==================== main.cu ====================
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <sys/stat.h> 

#include "cloth.cuh"
#include "integrator.cuh"
#include "params.cuh"
#include "applied_force.cuh"
#include "energy.cuh"
#include "convergence.cuh"
#include "extract_positions.cuh"
#include "renderer.h"
#include "vtk_export.h"

#define BLOCK_SIZE 128
#define MAX_STEPS 10000
#define CONVERGENCE_CHECK_INTERVAL 50

// Convergence thresholds
#define VELOCITY_THRESHOLD 0.1f
#define FORCE_THRESHOLD 0.1f

int main() {
    mkdir("output", 0777);
    std::string outputPath = "../output/initial_cloth.vtk";
    FILE* forceLog = fopen("../output/force_log_patch.csv", "w");
    fprintf(forceLog, "step");

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            fprintf(forceLog, ",Fy(%+d,%+d),Vy(%+d,%+d)", dx, dy, dx, dy);
        }
    }
    fprintf(forceLog, "\n");

    // CUDA device diagnostics
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess || deviceCount == 0) {
        printf("CUDA device error: %s\n", cudaGetErrorString(error_id));
        return EXIT_FAILURE;
    }
    printf("Found %d CUDA devices\n", deviceCount);

    // Simulation parameters
    const int h_num_x = 100;
    const int h_num_y = 100;
    const int total_nodes = h_num_x * h_num_y;
    const float clothWidth = 2.0f;
    const float clothHeight = 2.0f;
    const float force_strength = -40.0f;
    const int radius = 4;
    const int start_step = 0;
    const int end_step = 500;
    const std::string selectedMode = "gaussian";

    // Allocate and initialize cloth nodes
    ClothNode* d_nodes;
    cudaMalloc(&d_nodes, total_nodes * sizeof(ClothNode));
    initializeClothGrid(d_nodes, h_num_x, h_num_y, clothWidth, clothHeight);
    allocateExternalForceBuffer();
    uploadSimParamsToDevice(h_num_x, h_num_y, clothWidth, clothHeight);

    const float dx_val = host_dx;
    const float dy_val = host_dy;
    const float mass_val = host_node_mass;

    // Initialize springs
    std::vector<Spring> springs;
    for (int y = 0; y < h_num_y; y++) {
        for (int x = 0; x < h_num_x; x++) {
            const int idx = y * h_num_x + x;

            // Structural
            if (x < h_num_x - 1)
                springs.push_back({idx, idx + 1, dx_val, STRUCTURAL});
            if (y < h_num_y - 1)
                springs.push_back({idx, idx + h_num_x, dy_val, STRUCTURAL});

            // Shear
            const float diag = sqrtf(dx_val * dx_val + dy_val * dy_val);
            if (x < h_num_x - 1 && y < h_num_y - 1)
                springs.push_back({idx, idx + h_num_x + 1, diag, SHEAR});
            if (x > 0 && y < h_num_y - 1)
                springs.push_back({idx, idx + h_num_x - 1, diag, SHEAR});

            // Bending
            if (x < h_num_x - 2)
                springs.push_back({idx, idx + 2, 2.0f * dx_val, BENDING});
            if (y < h_num_y - 2)
                springs.push_back({idx, idx + 2 * h_num_x, 2.0f * dy_val, BENDING});
        }
    }

    Spring* d_springs;
    const int numSprings = springs.size();
    cudaMalloc(&d_springs, numSprings * sizeof(Spring));
    cudaMemcpy(d_springs, springs.data(), numSprings * sizeof(Spring), cudaMemcpyHostToDevice);
    const int springBlocks = (numSprings + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float3* d_old_accel;
    cudaMalloc(&d_old_accel, total_nodes * sizeof(float3));
    cudaMemset(d_old_accel, 0, total_nodes * sizeof(float3));

    float3* d_clothPositions;
    cudaMalloc(&d_clothPositions, total_nodes * sizeof(float3));
    cudaMemset(d_clothPositions, 0, total_nodes * sizeof(float3));

    ConvergenceData* d_conv_data;
    cudaMalloc(&d_conv_data, sizeof(ConvergenceData));

    float* d_kinetic_energy;
    float* d_potential_energy;
    cudaMalloc(&d_kinetic_energy, sizeof(float));
    cudaMalloc(&d_potential_energy, sizeof(float));

    const int threads = BLOCK_SIZE;
    const int blocks = (total_nodes + threads - 1) / threads;

    float h_dt;
    cudaMemcpyFromSymbol(&h_dt, dt, sizeof(float), 0, cudaMemcpyDeviceToHost);

    initRenderer(h_num_x, h_num_y);

    extractPositions<<<blocks, threads>>>(d_nodes, d_clothPositions, total_nodes);
    cudaDeviceSynchronize();

    for (int i = 0; i < 60 && !windowShouldClose(); ++i) {
        beginFrame();
        renderCloth(-1, d_clothPositions);
        endFrame();
    }

    std::vector<ClothNode> h_nodes(total_nodes);
    cudaMemcpy(h_nodes.data(), d_nodes, total_nodes * sizeof(ClothNode), cudaMemcpyDeviceToHost);
    writeClothToVTK(outputPath, h_nodes, springs, h_num_x, h_num_y);

    printf("Starting simulation...\n");

    bool converged = false;
    for (int step = 0; step <= MAX_STEPS && !converged; ++step) {
        // Integration step
        verletUpdatePosition<<<blocks, threads>>>(d_nodes, d_old_accel, h_dt);
        applyPinningConstraints<<<blocks, threads>>>(d_nodes);

        resetClothForces<<<blocks, threads>>>(d_nodes);
        applyGravity<<<blocks, threads>>>(d_nodes);
        applyViscousDamping<<<blocks, threads>>>(d_nodes);

        kernelResetExternalForces<<<blocks, threads>>>(d_externalForce, total_nodes);
        applyPrescribedForce(step, d_externalForce, h_num_x, h_num_y,
                             selectedMode, force_strength, radius,
                             start_step, end_step);
        kernelInjectForces<<<blocks, threads>>>(d_nodes, d_externalForce, total_nodes);

        applySpringForces<<<springBlocks, threads>>>(d_nodes, d_springs, numSprings);

        verletUpdateVelocity<<<blocks, threads>>>(d_nodes, d_old_accel, h_dt, (1.0f / mass_val));
        storeAcceleration<<<blocks, threads>>>(d_nodes, d_old_accel, 1.0f / mass_val);
        applyPinningConstraints<<<blocks, threads>>>(d_nodes);

        if (step % CONVERGENCE_CHECK_INTERVAL == 0 && step > 0) {
            resetConvergenceData<<<1, 1>>>(d_conv_data);
            cudaDeviceSynchronize();
            computeConvergenceMetrics<<<blocks, threads>>>(d_nodes, d_conv_data);
            cudaDeviceSynchronize();

            converged = checkConvergence(d_conv_data, total_nodes, VELOCITY_THRESHOLD, FORCE_THRESHOLD);

            if (converged) {
                printf("Simulation converged at step %d\n", step);
                break;
            }

            if (step % (CONVERGENCE_CHECK_INTERVAL * 10) == 0) {
                ConvergenceData h_conv;
                cudaMemcpy(&h_conv, d_conv_data, sizeof(ConvergenceData), cudaMemcpyDeviceToHost);
                printf("Step %d: Max velocity = %.6f, Max force = %.6f\n", 
                       step, h_conv.max_velocity, h_conv.max_force);
            }

            if (step % 50 == 0) {
                cudaMemcpy(h_nodes.data(), d_nodes, total_nodes * sizeof(ClothNode), cudaMemcpyDeviceToHost);

                int cx = h_num_x / 2;
                int cy = h_num_y / 2;

                printf("Step %d:\n", step);
                fprintf(forceLog, "%d", step);

                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int x = cx + dx;
                        int y = cy + dy;
                        int idx = y * h_num_x + x;

                        float fy = h_nodes[idx].force.y;
                        float vy = h_nodes[idx].vel.y;

                        printf("  Node (%d,%d): Vy = %.6f, Fy = %.6f\n", x, y, vy, fy);
                        fprintf(forceLog, ",%.6f,%.6f", fy, vy);
                    }
                }
                fprintf(forceLog, "\n");
            }
        }

        beginFrame();
        extractPositions<<<blocks, threads>>>(d_nodes, d_clothPositions, total_nodes);
        cudaDeviceSynchronize();
        renderCloth(step, d_clothPositions);

        if (step % 100 == 0) {
            std::vector<ClothNode> host_nodes(total_nodes);
            cudaMemcpy(host_nodes.data(), d_nodes, total_nodes * sizeof(ClothNode), cudaMemcpyDeviceToHost);
            char filename[256];
            sprintf(filename, "../output/cloth_step_%04d.vtk", step);
            writeClothToVTK(filename, host_nodes, springs, h_num_x, h_num_y);
        }

        endFrame();
    }

    if (converged)
        printf("Simulation reached equilibrium!\n");
    else
        printf("Simulation completed maximum steps without convergence.\n");

    cudaFree(d_nodes);
    cudaFree(d_springs);
    cudaFree(d_old_accel);
    cudaFree(d_clothPositions);
    cudaFree(d_conv_data);
    cudaFree(d_kinetic_energy);
    cudaFree(d_potential_energy);
    freeExternalForceBuffer();
    cleanupRenderer();
    fclose(forceLog);

    return 0;
}
