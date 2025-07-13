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
#include "spatial_hash.cuh"
#include "marble.cuh"

#define BLOCK_SIZE 128
#define MAX_STEPS 10000
#define CONVERGENCE_CHECK_INTERVAL 50

// Convergence thresholds
#define VELOCITY_THRESHOLD 0.1f
#define FORCE_THRESHOLD 0.1f

__host__ float getMaxVelocity(ClothNode* nodes, Marble* marbles) {
    return 0.1f;
}

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
    const float force_strength = 0.0f;
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


    extractPositions<<<blocks, threads>>>(d_nodes, d_clothPositions, total_nodes);
    cudaDeviceSynchronize();

    std::vector<ClothNode> h_nodes(total_nodes);
    cudaMemcpy(h_nodes.data(), d_nodes, total_nodes * sizeof(ClothNode), cudaMemcpyDeviceToHost);
    writeClothToVTK(outputPath, h_nodes, springs, h_num_x, h_num_y);

    float h_grid_cell_size;
    cudaMemcpyFromSymbol(&h_grid_cell_size, grid_cell_size, sizeof(float));

    float3 grid_min = make_float3(-clothWidth/2, -clothHeight/2, -1.0f);
    float3 grid_max = make_float3(clothWidth/2, clothHeight/2, 1.0f);
    int3 grid_dims = make_int3(
        ceil((grid_max.x - grid_min.x) / h_grid_cell_size),
        ceil((grid_max.y - grid_min.y) / h_grid_cell_size),
        ceil((grid_max.z - grid_min.z) / h_grid_cell_size)
    );
    int grid_size = grid_dims.x * grid_dims.y * grid_dims.z;

    // Allocate linked list structures
    int* d_cellHead;
    int* d_nodeNext;
    cudaMalloc(&d_cellHead, grid_size * sizeof(int));
    cudaMalloc(&d_nodeNext, total_nodes * sizeof(int));

    // Initialize marbles
    const int h_num_marbles = 10;
    Marble* h_marbles = new Marble[h_num_marbles];
    Marble* d_marbles;
    cudaMalloc(&d_marbles, h_num_marbles * sizeof(Marble));
    int marbleThreads = BLOCK_SIZE;
    int marbleBlocks = (h_num_marbles + marbleThreads - 1) / marbleThreads;
    
    float h_marble_radius_min, h_marble_radius_max, h_marble_density;
    cudaMemcpyFromSymbol(&h_marble_radius_min, marble_radius_min, sizeof(float));
    cudaMemcpyFromSymbol(&h_marble_radius_max, marble_radius_max, sizeof(float));
    cudaMemcpyFromSymbol(&h_marble_density, marble_density, sizeof(float));
    
    // DECLARE VARIABLES BEFORE ALLOCATING
    int* d_cellHeadMarble = nullptr;
    int* d_marbleNext = nullptr;
    cudaMalloc(&d_cellHeadMarble, grid_size * sizeof(int));
    cudaMalloc(&d_marbleNext, h_num_marbles * sizeof(int));
    
    initializeMarbles(h_marbles, h_num_marbles, h_marble_radius_min, 
                     h_marble_radius_max, h_marble_density);
    cudaMemcpy(d_marbles, h_marbles, h_num_marbles * sizeof(Marble), 
              cudaMemcpyHostToDevice);

    // Spatial grid for marbles
    int* d_marbleCellHead = nullptr;
    int* d_marbleNodeNext = nullptr;
    cudaMalloc(&d_marbleCellHead, grid_size * sizeof(int));
    cudaMalloc(&d_marbleNodeNext, h_num_marbles * sizeof(int));

    initRenderer(h_num_x, h_num_y, h_num_marbles);
    for (int i = 0; i < 60 && !windowShouldClose(); ++i) {
        beginFrame();
        renderClothAndMarbles(-1, d_clothPositions, nullptr, 0);
        endFrame();
    }

    printf("Starting simulation...\n");

    bool converged = false;
    const float base_dt = 0.001f;
    float adaptive_dt = base_dt;

    for (int step = 0; step <= MAX_STEPS && !converged; ++step) {
        float max_vel = getMaxVelocity(d_nodes, d_marbles);
        adaptive_dt = base_dt / (1.0f + max_vel * 0.1f); // Reduced sensitivity
        float current_dt = fminf(adaptive_dt, 0.005f); // Cap maximum dt

        // PHASE 1: RESET FORCES
        resetClothForces<<<blocks, threads>>>(d_nodes);
        resetMarbleForces<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles);
        cudaDeviceSynchronize();

        // PHASE 2: APPLY BASIC FORCES
        applyGravity<<<blocks, threads>>>(d_nodes);
        applyGravityToMarbles<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles);
        applySpringForces<<<springBlocks, threads>>>(d_nodes, d_springs, numSprings);
        applyViscousDamping<<<blocks, threads>>>(d_nodes);
        cudaDeviceSynchronize();
        
        // PHASE 3: EXTERNAL FORCES
        kernelResetExternalForces<<<blocks, threads>>>(d_externalForce, total_nodes);
        applyPrescribedForce(step, d_externalForce, h_num_x, h_num_y, selectedMode, 
                            force_strength, radius, start_step, end_step);
        kernelInjectForces<<<blocks, threads>>>(d_nodes, d_externalForce, total_nodes);
        cudaDeviceSynchronize();
        
        // PHASE 4: SPATIAL HASHING AND SELF-COLLISIONS
        // Cloth spatial hashing
        cudaMemset(d_cellHead, -1, grid_size * sizeof(int));
        extractPositions<<<blocks, threads>>>(d_nodes, d_clothPositions, total_nodes);
        buildSpatialLinkedList<<<blocks, threads>>>(d_clothPositions, d_cellHead, d_nodeNext, 
                                                total_nodes, grid_dims, grid_min, h_grid_cell_size);
        applyClothSelfContact<<<blocks, threads>>>(d_nodes, d_cellHead, d_nodeNext, 
                                                grid_dims, grid_min, h_grid_cell_size);
        cudaDeviceSynchronize();
        
        // Marble spatial hashing
        cudaMemset(d_marbleCellHead, -1, grid_size * sizeof(int));
        insertMarblesToGrid<<<marbleBlocks, marbleThreads>>>(d_marbles, d_marbleCellHead, 
                                                            d_marbleNodeNext, h_num_marbles, 
                                                            grid_dims, grid_min, h_grid_cell_size);
        cudaDeviceSynchronize();

        // PHASE 5: INTER-OBJECT COLLISIONS
        // Marble-marble interactions
        computeMarbleMarbleInteraction<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, 
                                                                    d_marbleCellHead, d_marbleNodeNext, 
                                                                    grid_dims, grid_min, h_grid_cell_size);
        
        // FIXED: Improved cloth-marble interaction (both methods)
        computeMarbleClothInteraction<<<blocks, threads>>>(d_nodes, d_marbles, h_num_marbles, current_dt);
        computeMarbleTriangleCollision<<<marbleBlocks, marbleThreads>>>(d_nodes, d_marbles, h_num_marbles);
        cudaDeviceSynchronize();
        
        // PHASE 6: INTEGRATION WITH STABILITY CHECKS
        // Cloth integration
        eulerIntegrateCloth<<<blocks, threads>>>(d_nodes, current_dt, 1.0f/mass_val);
        cudaDeviceSynchronize();
        
        // Marble integration with stability controls
        updateMarbleVelocities<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, current_dt);
        
        // FIXED: Apply constraints before position update
        float3 domainMin = make_float3(-2.5f, -0.5f, -2.5f);
        float3 domainMax = make_float3(2.5f, 8.0f, 2.5f);
        
        applyBoundaryConstraints<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, domainMin, domainMax);
        
        // Limit velocities for stability
        float maxVelocity = 15.0f;
        float maxAngularVel = 10.0f;
        limitMarbleVelocities<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, maxVelocity, maxAngularVel);
        
        // Update positions
        updateMarblePositionsAndOrientations<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, current_dt);
        
        // Apply energy dissipation for long-term stability
        float dampingFactor = 0.999f;
        applyEnergyDissipation<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, dampingFactor);
        
        cudaDeviceSynchronize();

        // PHASE 7: CONSTRAINTS
        applyPinningConstraints<<<blocks, threads>>>(d_nodes);
        cudaDeviceSynchronize();

        // PHASE 8: CONVERGENCE CHECK
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
                printf("Step %d: Max velocity = %.6f, Max force = %.6f, dt = %.6f\n", 
                    step, h_conv.max_velocity, h_conv.max_force, current_dt);
            }

            // FIXED: Additional stability monitoring
            if (step % 50 == 0) {
                // Check for any marbles that might have unstable behavior
                std::vector<Marble> h_marbles_check(h_num_marbles);
                cudaMemcpy(h_marbles_check.data(), d_marbles, h_num_marbles * sizeof(Marble), cudaMemcpyDeviceToHost);
                
                bool unstable = false;
                for (int i = 0; i < h_num_marbles; i++) {
                    float speed = sqrtf(h_marbles_check[i].vel.x * h_marbles_check[i].vel.x + 
                                    h_marbles_check[i].vel.y * h_marbles_check[i].vel.y + 
                                    h_marbles_check[i].vel.z * h_marbles_check[i].vel.z);
                    if (speed > 50.0f || h_marbles_check[i].pos.y < -10.0f || h_marbles_check[i].pos.y > 20.0f) {
                        printf("Warning: Marble %d unstable - speed: %.2f, pos: (%.2f, %.2f, %.2f)\n", 
                            i, speed, h_marbles_check[i].pos.x, h_marbles_check[i].pos.y, h_marbles_check[i].pos.z);
                        unstable = true;
                    }
                }
                
                if (unstable) {
                    printf("Applying emergency stabilization...\n");
                    // Apply emergency damping
                    float emergency_damping = 0.8f;
                    applyEnergyDissipation<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, emergency_damping);
                    cudaDeviceSynchronize();
                }
            }
        }

        // PHASE 9: RENDERING
        beginFrame();
        extractPositions<<<blocks, threads>>>(d_nodes, d_clothPositions, total_nodes);
        cudaDeviceSynchronize();
        renderClothAndMarbles(step, d_clothPositions, d_marbles, h_num_marbles);

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
    cudaFree(d_cellHead);
    cudaFree(d_nodeNext);
    freeExternalForceBuffer();
    cleanupRenderer();
    fclose(forceLog);

    return 0;
}
