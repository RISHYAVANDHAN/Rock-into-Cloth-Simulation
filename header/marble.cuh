#ifndef MARBLE_CUH
#define MARBLE_CUH

#pragma once

#include "float3_utils.cuh"

void initializeMarbles(Marble* marbles, int numMarbles, float min_radius, float max_radius, float density);

__global__ void resetMarbleForces(Marble* marbles, int numMarbles);

__global__ void updateMarbleVelocities(Marble* marbles, int numMarbles, float dt);

__global__ void updateMarblePositionsAndOrientations(Marble* marbles, int numMarbles, float dt);

__global__ void computeMarbleMarbleInteraction(Marble* marbles, int numMarbles, int* cellHead, int* marbleNodeNext, int3 gridSize, float3 gridMin, float cellSize);

__global__ void extractMarblePositions(Marble* marbles, float3* positions, int numMarbles);  

__global__ void applyGravityToMarbles(Marble* marbles, int numMarbles);

__global__ void applyBoundaryConstraints(Marble* marbles, int numMarbles, float3 domainMin, float3 domainMax);

__global__ void limitMarbleVelocities(Marble* marbles, int numMarbles, float maxVel, float maxAngVel);

__global__ void applyEnergyDissipation(Marble* marbles, int numMarbles, float dampingFactor);

__global__ void storeMarblePreviousPositions(Marble* marbles, int numMarbles);

__global__ void limitMarbleVelocities(Marble* marbles, int numMarbles, float maxVel, float maxAngVel);

__global__ void continuousMarbleClothCollision(Marble* marbles, int numMarbles, ClothNode* nodes, const int* cellHead, const int* nodeNext, int3 gridSize, float3 gridMin, float cellSize, float dt);

#endif