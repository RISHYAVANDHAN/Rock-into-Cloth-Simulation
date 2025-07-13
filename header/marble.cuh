#ifndef MARBLE_CUH
#define MARBLE_CUH

#include "float3_utils.cuh"

struct Marble {
    float3 pos;
    float3 vel;
    float3 force;
    float3 angular_vel;
    float3 torque;
    float4 orientation; // Quaternion (x,y,z,w)
    float radius;
    float mass;
    float inertia; // I = (2/5)*mass*radius²
};

void initializeMarbles(Marble* marbles, int numMarbles, float min_radius, float max_radius, float density);

__global__ void resetMarbleForces(Marble* marbles, int numMarbles);

__global__ void updateMarbleVelocities(Marble* marbles, int numMarbles, float dt);

__global__ void updateMarblePositionsAndOrientations(Marble* marbles, int numMarbles, float dt);

__global__ void computeMarbleMarbleInteraction(Marble* marbles, int numMarbles, int* cellHead, int* nodeNext, int3 gridSize, float3 gridMin, float cellSize);

__global__ void extractMarblePositions(Marble* marbles, float3* positions, int numMarbles);  

__global__ void applyGravityToMarbles(Marble* marbles, int numMarbles);

__global__ void applyBoundaryConstraints(Marble* marbles, int numMarbles, float3 domainMin, float3 domainMax);

__global__ void limitMarbleVelocities(Marble* marbles, int numMarbles, float maxVel, float maxAngVel);

__global__ void applyEnergyDissipation(Marble* marbles, int numMarbles, float dampingFactor);

#endif