#ifndef ENERGY_CUH
#define ENERGY_CUH

#include <cuda_runtime.h>
#include "cloth.cuh"
#include "params.cuh"

struct EnergyData {
    float total;
    float kinetic;
    float potential_grav;
    float potential_spring;  // Added spring energy
};

__global__ void resetEnergyBuffers(float* kinetic_energy, float* potential_energy);

__global__ void computeKineticEnergy(ClothNode* nodes, float* kinetic_energy);

__global__ void computePotentialEnergy(ClothNode* nodes, Spring* springs, int numSprings, float* potential_energy);

#endif