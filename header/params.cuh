#ifndef PARAMS_CUH
#define PARAMS_CUH

#include <cuda_runtime.h>

extern __constant__ float ks_structural;
extern __constant__ float ks_shear;
extern __constant__ float ks_bend;
extern __constant__ float kd;

extern __constant__ float node_mass;
extern __constant__ float dt;
extern __constant__ float3 gravity;

extern __constant__ float cloth_width;
extern __constant__ float cloth_height;
extern __constant__ int num_x;
extern __constant__ int num_y;
extern __constant__ float dx;
extern __constant__ float dy;
extern __constant__ float plastic_threshold;

extern float h_dt;
extern float3 h_gravity;
extern float host_dx;
extern float host_dy;
extern float host_node_mass;

extern __constant__ float contact_radius;
extern __constant__ float kn_contact;
extern __constant__ float kd_contact;
extern __constant__ float friction_mu;
extern __constant__ float grid_cell_size;

extern __constant__ float kn_marble;
extern __constant__ float kd_marble;
extern __constant__ float friction_mu_marble;

extern __constant__ int num_marbles;
extern __constant__ float marble_radius_min;
extern __constant__ float marble_radius_max;
extern __constant__ float marble_density;
extern __constant__ float kn_marble_marble;
extern __constant__ float kd_marble_marble;
extern __constant__ float friction_mu_marble_marble;

extern __constant__ float3 domain_min;
extern __constant__ float3 domain_max;

void uploadSimParamsToDevice(int Nx, int Ny, float width, float height);

#endif
