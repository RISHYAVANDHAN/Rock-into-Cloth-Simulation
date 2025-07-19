// ==================== main.cu (FIXED) ====================
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <sys/stat.h> 

#include "cloth.cuh"
#include "integrator.cuh"
#include "params.cuh"
#include "extract_positions.cuh"
#include "renderer.h"
#include "spatial_hash.cuh"
#include "marble.cuh"

#define BLOCK_SIZE 128
#define MAX_STEPS 10000
#define CONVERGENCE_CHECK_INTERVAL 50

int main() {

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

    // Allocate and initialize cloth nodes
    ClothNode* d_nodes = nullptr;
    cudaMalloc(&d_nodes, total_nodes * sizeof(ClothNode));
    initializeClothGrid(d_nodes, h_num_x, h_num_y, clothWidth, clothHeight);
    uploadSimParamsToDevice(h_num_x, h_num_y, clothWidth, clothHeight);

    float host_dx, host_dy, host_node_mass;
    cudaMemcpyFromSymbol(&host_dx, dx, sizeof(float), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_dy, dy, sizeof(float), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&host_node_mass, node_mass, sizeof(float), 0, cudaMemcpyDeviceToHost);

    // Initialize springs
    std::vector<Spring> springs;
    for (int y = 0; y < h_num_y; y++) {
        for (int x = 0; x < h_num_x; x++) {
            const int idx = y * h_num_x + x;

            // Structural
            if (x < h_num_x - 1)
                springs.push_back({idx, idx + 1, host_dx, STRUCTURAL});
            if (y < h_num_y - 1)
                springs.push_back({idx, idx + h_num_x, host_dy, STRUCTURAL});

            // Shear
            const float diag = sqrtf(host_dx * host_dx + host_dy * host_dy);
            if (x < h_num_x - 1 && y < h_num_y - 1)
                springs.push_back({idx, idx + h_num_x + 1, diag, SHEAR});
            if (x > 0 && y < h_num_y - 1)
                springs.push_back({idx, idx + h_num_x - 1, diag, SHEAR});

            // Bending
            if (x < h_num_x - 2)
                springs.push_back({idx, idx + 2, 2.0f * host_dx, BENDING});
            if (y < h_num_y - 2)
                springs.push_back({idx, idx + 2 * h_num_x, 2.0f * host_dy, BENDING});
        }
    }

    Spring* d_springs = nullptr;
    const int numSprings = springs.size();
    cudaMalloc(&d_springs, numSprings * sizeof(Spring));
    cudaMemcpy(d_springs, springs.data(), numSprings * sizeof(Spring), cudaMemcpyHostToDevice);
    const int springBlocks = (numSprings + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float3* d_clothPositions = nullptr;
    cudaMalloc(&d_clothPositions, total_nodes * sizeof(float3));
    cudaMemset(d_clothPositions, 0, total_nodes * sizeof(float3));

    // Initialize marbles
    const int h_num_marbles = 10;
    Marble* h_marbles = new Marble[h_num_marbles];
    Marble* d_marbles = nullptr;
    cudaMalloc(&d_marbles, h_num_marbles * sizeof(Marble));
    int marbleThreads = BLOCK_SIZE;
    int marbleBlocks = (h_num_marbles + marbleThreads - 1) / marbleThreads;
    
    float h_marble_radius_min, h_marble_radius_max, h_marble_density;
    cudaMemcpyFromSymbol(&h_marble_radius_min, marble_radius_min, sizeof(float));
    cudaMemcpyFromSymbol(&h_marble_radius_max, marble_radius_max, sizeof(float));
    cudaMemcpyFromSymbol(&h_marble_density, marble_density, sizeof(float));

    // Spatial grid setup - SINGLE GRID FOR ALL COLLISION DETECTION
    float h_grid_cell_size;
    cudaMemcpyFromSymbol(&h_grid_cell_size, grid_cell_size, sizeof(float));
    float3 grid_min = make_float3(
        -clothWidth/2 - 2*h_marble_radius_max, 
        -1.0f,
        -clothHeight/2 - 2*h_marble_radius_max
    );
    float3 grid_max = make_float3(
        clothWidth/2 + 2*h_marble_radius_max,
        10.0f + h_num_marbles*0.5f,
        clothHeight/2 + 2*h_marble_radius_max
    );
    int3 grid_dims = make_int3(
        ceil((grid_max.x - grid_min.x) / h_grid_cell_size),
        ceil((grid_max.y - grid_min.y) / h_grid_cell_size),
        ceil((grid_max.z - grid_min.z) / h_grid_cell_size)
    );
    int grid_size = grid_dims.x * grid_dims.y * grid_dims.z;
    
    // Allocate unified spatial grid structures
    int* d_cellHead = nullptr;
    int* d_nodeNext = nullptr;
    cudaMalloc(&d_cellHead, grid_size * sizeof(int));
    cudaMalloc(&d_nodeNext, total_nodes * sizeof(int));
    
    // Separate marble grid for marble-marble collisions
    int* d_marbleCellHead = nullptr;
    int* d_marbleNodeNext = nullptr;
    cudaMalloc(&d_marbleCellHead, grid_size * sizeof(int));
    cudaMalloc(&d_marbleNodeNext, h_num_marbles * sizeof(int));

    // Initialize marbles on host and copy to device
    initializeMarbles(h_marbles, h_num_marbles, h_marble_radius_min, 
                     h_marble_radius_max, h_marble_density);
    cudaMemcpy(d_marbles, h_marbles, h_num_marbles * sizeof(Marble), 
              cudaMemcpyHostToDevice);

    // Initialize renderer
    initRenderer(h_num_x, h_num_y, h_num_marbles);
    for (int i = 0; i < 60 && !windowShouldClose(); ++i) {
        beginFrame();
        renderClothAndMarbles(-1, d_clothPositions, nullptr, 0);
        endFrame();
    }

    printf("Starting simulation...\n");

    const float dt = 0.001f;
    const int threads = BLOCK_SIZE;
    const int blocks = (total_nodes + threads - 1) / threads;

    #define NUM_SUBSTEPS 2
    #define MAX_MARBLE_VELOCITY 10.0f
    #define MAX_CLOTH_NODE_VELOCITY 8.0f
    #define MIN_TIMESTEP 0.0001f
    #define MAX_TIMESTEP 0.002f

    for (int step = 0; step <= MAX_STEPS && !windowShouldClose(); ++step) {
        
        // SUBCYCLING: Multiple small timesteps per frame
        for (int substep = 0; substep < NUM_SUBSTEPS; substep++) {
            
            // Store previous positions for continuous collision detection
            storeMarblePreviousPositions<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles);
            cudaDeviceSynchronize();
            
            // PHASE 1: RESET FORCES
            resetClothForces<<<blocks, threads>>>(d_nodes, total_nodes);
            resetMarbleForces<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles);
            cudaDeviceSynchronize();

            // PHASE 2: APPLY BASIC FORCES
            applyGravity<<<blocks, threads>>>(d_nodes);
            applyGravityToMarbles<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles);
            applySpringForces<<<springBlocks, threads>>>(d_nodes, d_springs, numSprings);
            applyViscousDamping<<<blocks, threads>>>(d_nodes);
            cudaDeviceSynchronize();
            
            // PHASE 3: BUILD SPATIAL GRIDS
            cudaMemset(d_cellHead, -1, grid_size * sizeof(int));
            extractPositions<<<blocks, threads>>>(d_nodes, d_clothPositions, total_nodes);
            buildSpatialLinkedList<<<blocks, threads>>>(d_clothPositions, d_cellHead, d_nodeNext, 
                                                    total_nodes, grid_dims, grid_min, h_grid_cell_size);
            cudaDeviceSynchronize();

            cudaMemset(d_marbleCellHead, -1, grid_size * sizeof(int));
            insertMarblesToGrid<<<marbleBlocks, marbleThreads>>>(d_marbles, d_marbleCellHead, 
                                                                d_marbleNodeNext, h_num_marbles, 
                                                                grid_dims, grid_min, h_grid_cell_size);
            cudaDeviceSynchronize();
            
            // PHASE 4: COLLISION DETECTION & RESPONSE
            applyClothSelfContact<<<blocks, threads>>>(d_nodes, d_cellHead, d_nodeNext, 
                                                    grid_dims, grid_min, h_grid_cell_size);
            
            continuousMarbleClothCollision<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, d_nodes,
                                                                        d_cellHead, d_nodeNext,
                                                                        grid_dims, grid_min, h_grid_cell_size, dt);
            
            computeMarbleMarbleInteraction<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, 
                                                                        d_marbleCellHead, d_marbleNodeNext, 
                                                                        grid_dims, grid_min, h_grid_cell_size);
            cudaDeviceSynchronize();
            
            // PHASE 5: VELOCITY LIMITING (Before Integration)
            limitMarbleVelocities<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, 
                                                                MAX_MARBLE_VELOCITY, 10.0f);
            limitClothNodeVelocities<<<blocks, threads>>>(d_nodes, total_nodes, MAX_CLOTH_NODE_VELOCITY);
            cudaDeviceSynchronize();
            
            // PHASE 6: INTEGRATION
            eulerIntegrateCloth<<<blocks, threads>>>(d_nodes, dt, 1.0f/host_node_mass);
            updateMarbleVelocities<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, dt);
            
            // Boundary constraints
            float3 domainMin = make_float3(-2.5f, -0.5f, -2.5f);
            float3 domainMax = make_float3(2.5f, 8.0f, 2.5f);
            applyBoundaryConstraints<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, domainMin, domainMax);
            
            updateMarblePositionsAndOrientations<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, dt);
            
            // Energy dissipation (lighter for substeps)
            float substep_damping = powf(0.999f, 1.0f/NUM_SUBSTEPS);
            applyEnergyDissipation<<<marbleBlocks, marbleThreads>>>(d_marbles, h_num_marbles, substep_damping);
            cudaDeviceSynchronize();

            // PHASE 7: CONSTRAINTS
            applyPinningConstraints<<<blocks, threads>>>(d_nodes);
            cudaDeviceSynchronize();
        }

        // PHASE 8: RENDERING (Only once per main timestep)
        beginFrame();
        extractPositions<<<blocks, threads>>>(d_nodes, d_clothPositions, total_nodes);
        cudaDeviceSynchronize();
        renderClothAndMarbles(step, d_clothPositions, d_marbles, h_num_marbles);
        endFrame();
    }

    // Cleanup
    cudaFree(d_nodes);
    cudaFree(d_springs);
    cudaFree(d_clothPositions);
    cudaFree(d_cellHead);
    cudaFree(d_nodeNext);
    cudaFree(d_marbleCellHead);
    cudaFree(d_marbleNodeNext);
    cudaFree(d_marbles);
    cleanupRenderer();
    delete[] h_marbles;

    return 0;
}