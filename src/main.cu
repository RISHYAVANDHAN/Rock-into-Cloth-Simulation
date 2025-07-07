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
#define VELOCITY_THRESHOLD 0.01f
#define FORCE_THRESHOLD 0.1f

int main() {
    mkdir("output", 0777);
    std::string outputPath = "../output/initial_cloth.vtk";

    // CUDA device diagnostics
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
               static_cast<int>(error_id), cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found\n");
        exit(EXIT_FAILURE);
    }

    printf("Found %d CUDA devices\n", deviceCount);

    // Simulation parameters
    const int h_num_x = 100;
    const int h_num_y = 100;
    const int total_nodes = h_num_x * h_num_y;
    const float clothWidth = 2.0f;
    const float clothHeight = 2.0f;
    const float force_strength = 30.0f;
    const int radius = 4;
    const int start_step = 0;
    const int end_step = 500;
    const std::string selectedMode = "patch";

    // Device memory allocation
    ClothNode* d_nodes;
    cudaMalloc(&d_nodes, total_nodes * sizeof(ClothNode));
    initializeClothGrid(d_nodes, h_num_x, h_num_y, clothWidth, clothHeight);
    allocateExternalForceBuffer();

    // Initialize simulation parameters
    uploadSimParamsToDevice(h_num_x, h_num_y, clothWidth, clothHeight);
    
    // Get host-accessible parameters
    const float dx_val = host_dx;
    const float dy_val = host_dy;
    const float mass_val = host_node_mass;

    // Spring initialization
    std::vector<Spring> springs;
    for (int y = 0; y < h_num_y; y++) {
        for (int x = 0; x < h_num_x; x++) {
            const int idx = y * h_num_x + x;
            
            // Structural springs
            if (x < h_num_x - 1) 
                springs.push_back({idx, idx + 1, dx_val});
            if (y < h_num_y - 1) 
                springs.push_back({idx, idx + h_num_x, dy_val});
            
            // Diagonal springs
            const float diag = sqrtf(dx_val*dx_val + dy_val*dy_val);
            if (x < h_num_x - 1 && y < h_num_y - 1)
                springs.push_back({idx, idx + h_num_x + 1, diag});
            if (x > 0 && y < h_num_y - 1)
                springs.push_back({idx, idx + h_num_x - 1, diag});
        }
    }

    // Upload springs to device
    Spring* d_springs;
    const int numSprings = springs.size();
    cudaMalloc(&d_springs, numSprings * sizeof(Spring));
    cudaMemcpy(d_springs, springs.data(), numSprings * sizeof(Spring), cudaMemcpyHostToDevice);
    const int springBlocks = (numSprings + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Acceleration buffers
    float3* d_old_accel;
    cudaMalloc(&d_old_accel, total_nodes * sizeof(float3));
    cudaMemset(d_old_accel, 0, total_nodes * sizeof(float3));

    // Visualization buffers
    float3* d_clothPositions;
    cudaMalloc(&d_clothPositions, total_nodes * sizeof(float3));
    cudaMemset(d_clothPositions, 0, total_nodes * sizeof(float3));

    // Convergence data
    ConvergenceData* d_conv_data;
    cudaMalloc(&d_conv_data, sizeof(ConvergenceData));

    // Energy buffers
    float* d_kinetic_energy;
    float* d_potential_energy;
    cudaMalloc(&d_kinetic_energy, sizeof(float));
    cudaMalloc(&d_potential_energy, sizeof(float));

    // Kernel configuration
    const int threads = BLOCK_SIZE;
    const int blocks = (total_nodes + threads - 1) / threads;

    // Get dt from device
    float h_dt;
    cudaMemcpyFromSymbol(&h_dt, dt, sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Initialize renderer
    initRenderer(h_num_x, h_num_y);

    // Extract initial positions
    extractPositions<<<blocks, threads>>>(d_nodes, d_clothPositions, total_nodes);
    cudaDeviceSynchronize();

    // Render initial state
    for (int i = 0; i < 60 && !windowShouldClose(); ++i) {
        beginFrame();
        renderCloth(-1, d_clothPositions);
        endFrame();
    }

    // Copy nodes to host and export initial mesh
    std::vector<ClothNode> h_nodes(total_nodes);
    cudaMemcpy(h_nodes.data(), d_nodes, total_nodes * sizeof(ClothNode), cudaMemcpyDeviceToHost);
    writeClothToVTK(outputPath, h_nodes, springs, h_num_x, h_num_y);

    printf("Starting simulation...\n");

    // FIXED: Main simulation loop with convergence checking
    bool converged = false;
    for (int step = 0; step <= MAX_STEPS && !windowShouldClose() && !converged; ++step) {
        
        // Position update
        verletUpdatePosition<<<blocks, threads>>>(d_nodes, d_old_accel, h_dt);
        applyPinningConstraints<<<blocks, threads>>>(d_nodes);

        // Force computation
        resetClothForces<<<blocks, threads>>>(d_nodes);
        applyGravity<<<blocks, threads>>>(d_nodes);
        kernelResetExternalForces<<<blocks, threads>>>(d_externalForce, total_nodes);
        applyPrescribedForce(step, d_externalForce, h_num_x, h_num_y,
                            selectedMode, force_strength, radius,
                            start_step, end_step);
        kernelInjectForces<<<blocks, threads>>>(d_nodes, d_externalForce, total_nodes);
        applySpringForces<<<springBlocks, threads>>>(d_nodes, d_springs, numSprings);

        // Velocity update
        verletUpdateVelocity<<<blocks, threads>>>(d_nodes, d_old_accel, h_dt);
        storeAcceleration<<<blocks, threads>>>(d_nodes, d_old_accel, 1.0f / mass_val);
        applyPinningConstraints<<<blocks, threads>>>(d_nodes);

        // Check convergence every CONVERGENCE_CHECK_INTERVAL steps
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
            
            // Optional: Print convergence metrics
            if (step % (CONVERGENCE_CHECK_INTERVAL * 10) == 0) {
                ConvergenceData h_conv;
                cudaMemcpy(&h_conv, d_conv_data, sizeof(ConvergenceData), cudaMemcpyDeviceToHost);
                printf("Step %d: Max velocity = %.6f, Max force = %.6f\n", 
                       step, h_conv.max_velocity, h_conv.max_force);
            }
        }

        // Render every frame
        beginFrame();
        extractPositions<<<blocks, threads>>>(d_nodes, d_clothPositions, total_nodes);
        cudaDeviceSynchronize();
        renderCloth(step, d_clothPositions);
        
        // Export VTK files periodically
        if (step % 100 == 0) {
            std::vector<ClothNode> host_nodes(total_nodes);
            cudaMemcpy(host_nodes.data(), d_nodes, total_nodes * sizeof(ClothNode), cudaMemcpyDeviceToHost);

            char filename[256];
            sprintf(filename, "../output/cloth_step_%04d.vtk", step);
            writeClothToVTK(filename, host_nodes, springs, h_num_x, h_num_y);
        }
        
        endFrame();
    }

    if (converged) {
        printf("Simulation reached equilibrium!\n");
    } else {
        printf("Simulation completed maximum steps without convergence.\n");
    }

    // Cleanup
    cudaFree(d_nodes);
    cudaFree(d_springs);
    cudaFree(d_old_accel);
    cudaFree(d_clothPositions);
    cudaFree(d_conv_data);
    cudaFree(d_kinetic_energy);
    cudaFree(d_potential_energy);
    freeExternalForceBuffer();
    cleanupRenderer();
    
    return 0;
}