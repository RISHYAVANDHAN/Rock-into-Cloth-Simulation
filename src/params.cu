// Fixed params.cu with proper marble physics
#include "params.cuh"

// Host-side variables
float h_dt;
float3 h_gravity;
float host_dx;
float host_dy;
float host_node_mass;

// Device-side constants
__constant__ float ks_structural;
__constant__ float ks_shear;
__constant__ float ks_bend;
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
__constant__ float contact_radius;
__constant__ float kn_contact;
__constant__ float kd_contact;
__constant__ float friction_mu;
__constant__ float grid_cell_size;
__constant__ float kn_marble;
__constant__ float kd_marble;
__constant__ float friction_mu_marble;
__constant__ int num_marbles;
__constant__ float marble_radius_min;
__constant__ float marble_radius_max;
__constant__ float marble_density;
__constant__ float kn_marble_marble;
__constant__ float kd_marble_marble;
__constant__ float friction_mu_marble_marble;
__constant__ float3 domain_min;
__constant__ float3 domain_max;
__constant__ float boundary_restitution;
__constant__ float boundary_friction;
__constant__ float marble_grid_cell_size;

void uploadSimParamsToDevice(int Nx, int Ny, float width, float height) {
    // CLOTH PARAMETERS - Keep existing values
    float h_ks_structural = 10000.0f;
    float h_ks_shear = 10000.0f;
    float h_ks_bend = 3000.0f;
    float h_kd = 0.15f;
    float h_mass = 1.0f;
    
    // FIXED: Larger timestep for stability
    
    float h_dx_val = width / (Nx - 1);
    float h_dy_val = height / (Ny - 1);
    float h_threshold = 5.0f;
    float3 h_gravity_val = make_float3(0.0f, -9.81f, 0.0f);
    float h_contact_radius = 0.5f * min(h_dx_val, h_dy_val);
    float h_kn_contact = 5000.0f;
    float h_kd_contact = 0.0f;
    float h_friction_mu = 0.3f;
    float h_grid_cell_size = 2.5f * h_contact_radius;

    h_gravity = h_gravity_val;
    host_dx = h_dx_val;
    host_dy = h_dy_val;
    host_node_mass = h_mass;

    // MARBLE-CLOTH INTERACTION - Softer springs
    float h_kn_marble = 2000.0f;      
    float h_kd_marble = 200.0f;       
    float h_friction_mu_marble = 0.4f;

    int h_num_marbles = 5;
    float h_marble_radius_min = 0.08f;  
    float h_marble_radius_max = 0.15f;  
    float h_marble_density = 2.0f;
    
    // MARBLE-MARBLE INTERACTION - Much softer for stability
    float h_kn_marble_marble = 3000.0f;   
    float h_kd_marble_marble = 150.0f; 
    float h_friction_mu_marble_marble = 0.5f;

    // DOMAIN BOUNDARIES
    float3 h_domain_min = make_float3(-2.0f, -8.5f, -2.0f);
    float3 h_domain_max = make_float3(2.0f, 6.0f, 2.0f);
    float h_boundary_restitution = 0.6f;  
    float h_boundary_friction = 0.9f;     
    float h_marble_grid_size = 2.0f * h_marble_radius_max * 2.5f;

    // Copy all parameters to device
    cudaMemcpyToSymbol(marble_grid_cell_size, &h_marble_grid_size, sizeof(float));
    cudaMemcpyToSymbol(domain_min, &h_domain_min, sizeof(float3));
    cudaMemcpyToSymbol(domain_max, &h_domain_max, sizeof(float3));
    cudaMemcpyToSymbol(boundary_restitution, &h_boundary_restitution, sizeof(float));
    cudaMemcpyToSymbol(boundary_friction, &h_boundary_friction, sizeof(float));

    cudaMemcpyToSymbol(num_marbles, &h_num_marbles, sizeof(int));
    cudaMemcpyToSymbol(marble_radius_min, &h_marble_radius_min, sizeof(float));
    cudaMemcpyToSymbol(marble_radius_max, &h_marble_radius_max, sizeof(float));
    cudaMemcpyToSymbol(marble_density, &h_marble_density, sizeof(float));
    cudaMemcpyToSymbol(kn_marble_marble, &h_kn_marble_marble, sizeof(float));
    cudaMemcpyToSymbol(kd_marble_marble, &h_kd_marble_marble, sizeof(float));
    cudaMemcpyToSymbol(friction_mu_marble_marble, &h_friction_mu_marble_marble, sizeof(float));

    cudaMemcpyToSymbol(kn_marble, &h_kn_marble, sizeof(float));
    cudaMemcpyToSymbol(kd_marble, &h_kd_marble, sizeof(float));
    cudaMemcpyToSymbol(friction_mu_marble, &h_friction_mu_marble, sizeof(float));

    cudaMemcpyToSymbol(ks_structural, &h_ks_structural, sizeof(float));
    cudaMemcpyToSymbol(ks_shear, &h_ks_shear, sizeof(float));
    cudaMemcpyToSymbol(ks_bend, &h_ks_bend, sizeof(float));
    cudaMemcpyToSymbol(kd, &h_kd, sizeof(float));
    cudaMemcpyToSymbol(node_mass, &h_mass, sizeof(float));
    cudaMemcpyToSymbol(gravity, &h_gravity_val, sizeof(float3));
    cudaMemcpyToSymbol(cloth_width, &width, sizeof(float));
    cudaMemcpyToSymbol(cloth_height, &height, sizeof(float));
    cudaMemcpyToSymbol(num_x, &Nx, sizeof(int));
    cudaMemcpyToSymbol(num_y, &Ny, sizeof(int));
    cudaMemcpyToSymbol(dx, &h_dx_val, sizeof(float));
    cudaMemcpyToSymbol(dy, &h_dy_val, sizeof(float));
    cudaMemcpyToSymbol(plastic_threshold, &h_threshold, sizeof(float));
    cudaMemcpyToSymbol(contact_radius, &h_contact_radius, sizeof(float));
    cudaMemcpyToSymbol(kn_contact, &h_kn_contact, sizeof(float));
    cudaMemcpyToSymbol(kd_contact, &h_kd_contact, sizeof(float));
    cudaMemcpyToSymbol(friction_mu, &h_friction_mu, sizeof(float));
    cudaMemcpyToSymbol(grid_cell_size, &h_grid_cell_size, sizeof(float));
}