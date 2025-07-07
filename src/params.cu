// ==================== params.cu ====================
#include "params.cuh"

// Host-side variables
float h_dt;
float3 h_gravity;
float host_dx;
float host_dy;
float host_node_mass;

// Device-side constants
__constant__ float ks;
__constant__ float kd;
__constant__ float node_mass;
__constant__ float dt;
__constant__ float3 gravity;
__constant__ float cloth_width;
__constant__ float cloth_height;
__constant__ int num_x;
__constant__ int num_y;
__constant__ float dx;
__constant__ float dy;
__constant__ float plastic_threshold;

void uploadSimParamsToDevice(int Nx, int Ny, float width, float height) {
    float h_ks = 100.0f;        
    float h_kd = 2.0f;         
    float h_mass = 1.0f;
    float h_dt_val = 0.005f; // Smaller timestep for stability
    float h_dx_val = width / (Nx - 1);
    float h_dy_val = height / (Ny - 1);
    float h_threshold = 1.25f;
    float3 h_gravity_val = make_float3(0.0f, -9.81f, 0.0f);

    // Set host-side copies
    h_dt = h_dt_val;
    h_gravity = h_gravity_val;
    host_dx = h_dx_val;
    host_dy = h_dy_val;
    host_node_mass = h_mass;

    // Copy values to device constants
    cudaMemcpyToSymbol(ks, &h_ks, sizeof(float));
    cudaMemcpyToSymbol(kd, &h_kd, sizeof(float));
    cudaMemcpyToSymbol(node_mass, &h_mass, sizeof(float));
    cudaMemcpyToSymbol(dt, &h_dt_val, sizeof(float));
    cudaMemcpyToSymbol(gravity, &h_gravity_val, sizeof(float3));
    cudaMemcpyToSymbol(cloth_width, &width, sizeof(float));
    cudaMemcpyToSymbol(cloth_height, &height, sizeof(float));
    cudaMemcpyToSymbol(num_x, &Nx, sizeof(int));
    cudaMemcpyToSymbol(num_y, &Ny, sizeof(int));
    cudaMemcpyToSymbol(dx, &h_dx_val, sizeof(float));
    cudaMemcpyToSymbol(dy, &h_dy_val, sizeof(float));
    cudaMemcpyToSymbol(plastic_threshold, &h_threshold, sizeof(float));
}
