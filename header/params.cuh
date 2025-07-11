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

void uploadSimParamsToDevice(int Nx, int Ny, float width, float height);

#endif
