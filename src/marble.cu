// Fixed marble initialization and physics
#include "marble.cuh"
#include "spatial_hash.cuh"
#include "params.cuh"
#include <math.h>

__device__ __host__ float3 rotateVector(float3 v, float4 q) {
    float3 u = make_float3(q.x, q.y, q.z);
    float s = q.w;
    return 2.0f * dot(u, v) * u + (s*s - dot(u, u)) * v + 2.0f * s * cross(u, v);
}

__global__ void extractMarblePositions(Marble* marbles, float3* positions, int numMarbles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numMarbles) {
        positions[i] = marbles[i].pos;
    }
}

__global__ void storeMarblePreviousPositions(Marble* marbles, int numMarbles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numMarbles) return;
    marbles[i].prevPos = marbles[i].pos;
}

__global__ void limitMarbleVelocities(Marble* marbles, int numMarbles, float maxVel, float maxAngVel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numMarbles) return;
    
    float speed = length(marbles[i].vel);
    if (speed > maxVel) {
        marbles[i].vel = (maxVel / speed) * marbles[i].vel;
    }
    
    float angSpeed = length(marbles[i].angular_vel);
    if (angSpeed > maxAngVel) {
        marbles[i].angular_vel = (maxAngVel / angSpeed) * marbles[i].angular_vel;
    }
}

void initializeMarbles(Marble* marbles, int numMarbles, float min_radius, float max_radius, float density) {
    srand(42);
    
    for (int i = 0; i < numMarbles; ++i) {
        float radius = min_radius + (max_radius - min_radius) * (rand() / (float)RAND_MAX);
        marbles[i].radius = radius;
        marbles[i].mass = (4.0f/3.0f) * M_PI * radius*radius*radius * density;
        marbles[i].inertia = 0.4f * marbles[i].mass * radius*radius;
        
        float spawn_radius = 0.5f;
        float angle = (2.0f * M_PI * i) / numMarbles;
        float spawn_distance = spawn_radius * sqrtf(rand() / (float)RAND_MAX);
        
        marbles[i].pos = make_float3(
            spawn_distance * cosf(angle),
            4.0f + radius + i * 0.5f,
            spawn_distance * sinf(angle)
        );
        
        marbles[i].prevPos = marbles[i].pos; // Initialize previous position
        
        marbles[i].vel = make_float3(
            0.1f * (rand()/(float)RAND_MAX - 0.5f),
            -1.0f - i * 0.2f,
            0.1f * (rand()/(float)RAND_MAX - 0.5f)
        );
        
        marbles[i].angular_vel = make_float3(
            0.05f * (rand()/(float)RAND_MAX - 0.5f),
            0.05f * (rand()/(float)RAND_MAX - 0.5f),
            0.05f * (rand()/(float)RAND_MAX - 0.5f)
        );
        marbles[i].orientation = make_float4(0,0,0,1);
        marbles[i].force = make_float3(0,0,0);
        marbles[i].torque = make_float3(0,0,0);
    }
}

__global__ void resetMarbleForces(Marble* marbles, int numMarbles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numMarbles) return;
    marbles[idx].force = make_float3(0,0,0);
    marbles[idx].torque = make_float3(0,0,0);
}

__global__ void applyGravityToMarbles(Marble* marbles, int numMarbles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numMarbles) return;
    marbles[idx].force += marbles[idx].mass * gravity;
}

__global__ void updateMarbleVelocities(Marble* marbles, int numMarbles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numMarbles) return;
    
    // Add velocity limiting during integration
    float3 acceleration = marbles[idx].force / marbles[idx].mass;
    float3 angular_acceleration = marbles[idx].torque / marbles[idx].inertia;
    
    // Limit accelerations to prevent instability
    float max_accel = 50.0f;
    float accel_mag = length(acceleration);
    if (accel_mag > max_accel) {
        acceleration = (max_accel / accel_mag) * acceleration;
    }
    
    float max_ang_accel = 20.0f;
    float ang_accel_mag = length(angular_acceleration);
    if (ang_accel_mag > max_ang_accel) {
        angular_acceleration = (max_ang_accel / ang_accel_mag) * angular_acceleration;
    }
    
    marbles[idx].vel += acceleration * dt;
    marbles[idx].angular_vel += angular_acceleration * dt;
}

__global__ void updateMarblePositionsAndOrientations(Marble* marbles, int numMarbles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numMarbles) return;
    
    marbles[idx].pos += marbles[idx].vel * dt;
    
    // Quaternion integration
    float4 q = marbles[idx].orientation;
    float3 ω = marbles[idx].angular_vel;
    float4 dq = make_float4(
        0.5f * ( ω.x*q.w + ω.y*q.z - ω.z*q.y),
        0.5f * (-ω.x*q.z + ω.y*q.w + ω.z*q.x),
        0.5f * ( ω.x*q.y - ω.y*q.x + ω.z*q.w),
        0.5f * (-ω.x*q.x - ω.y*q.y - ω.z*q.z)
    );
    q.x += dq.x * dt; q.y += dq.y * dt; 
    q.z += dq.z * dt; q.w += dq.w * dt;
    
    float len = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    if (len > 1e-6f) {
        marbles[idx].orientation = make_float4(q.x/len, q.y/len, q.z/len, q.w/len);
    }
}

__global__ void computeMarbleMarbleInteraction(Marble* marbles, int numMarbles,int* cellHead, int* marbleNodeNext,int3 gridSize, float3 gridMin, float cellSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numMarbles) return;

    float3 pos_i = marbles[i].pos;
    
    // Use consistent grid position calculation
    int3 cell_coord = make_int3(
        floorf((pos_i.x - gridMin.x) / cellSize),
        floorf((pos_i.y - gridMin.y) / cellSize),
        floorf((pos_i.z - gridMin.z) / cellSize)
    );

    // Check neighboring cells
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int3 neighbor_cell = make_int3(
                    cell_coord.x + dx,
                    cell_coord.y + dy,
                    cell_coord.z + dz
                );

                if (neighbor_cell.x < 0 || neighbor_cell.x >= gridSize.x ||
                    neighbor_cell.y < 0 || neighbor_cell.y >= gridSize.y ||
                    neighbor_cell.z < 0 || neighbor_cell.z >= gridSize.z)
                    continue;
                // Use consistent hash calculation
                int hash = calcHash(neighbor_cell, gridSize);
                int j = cellHead[hash];
                // Traverse the linked list using the correct array
                while (j != -1) {
                    if (j != i && j > i) { // Avoid self-collision and double-processing
                        float3 pos_j = marbles[j].pos;
                        float3 delta = pos_j - pos_i;
                        float dist = length(delta);

                        float min_sep = marbles[i].radius + marbles[j].radius;
                        
                        if (dist < min_sep && dist > 1e-6f) {
                            float3 normal = delta / dist;
                            float overlap = min_sep - dist;
                            // Contact point and relative velocity calculation
                            float3 contact_pt = pos_i + normal * marbles[i].radius;
                            float3 r_i = contact_pt - pos_i;
                            float3 r_j = contact_pt - pos_j;

                            float3 v_i = marbles[i].vel + cross(marbles[i].angular_vel, r_i);
                            float3 v_j = marbles[j].vel + cross(marbles[j].angular_vel, r_j);
                            float3 v_rel = v_j - v_i;
                            float vn = dot(v_rel, normal);
                            // Force calculation (keep your existing values)
                            float repulsion = kn_marble_marble * overlap;
                            if (overlap > 0.02f * min_sep) {
                                float k = overlap / min_sep;
                                repulsion *= (1.0f + 4.0f * k);
                            }

                            float3 Fn = repulsion * normal - kd_marble * vn * normal;
                            // Apply forces with proper atomic operations
                            atomicAdd(&marbles[i].force.x, -Fn.x);
                            atomicAdd(&marbles[i].force.y, -Fn.y);
                            atomicAdd(&marbles[i].force.z, -Fn.z);

                            atomicAdd(&marbles[j].force.x, Fn.x);
                            atomicAdd(&marbles[j].force.y, Fn.y);
                            atomicAdd(&marbles[j].force.z, Fn.z);
                            // Apply torques
                            float3 tau_i = cross(r_i, -Fn);
                            float3 tau_j = cross(r_j, Fn);
                            atomicAdd(&marbles[i].torque.x, tau_i.x);
                            atomicAdd(&marbles[i].torque.y, tau_i.y);
                            atomicAdd(&marbles[i].torque.z, tau_i.z);
                            atomicAdd(&marbles[j].torque.x, tau_j.x);
                            atomicAdd(&marbles[j].torque.y, tau_j.y);
                            atomicAdd(&marbles[j].torque.z, tau_j.z);
                        }
                    }
                    j = marbleNodeNext[j];  // <- Use consistent array name
                }
            }
        }
    }
}

__global__ void applyBoundaryConstraints(Marble* marbles, int numMarbles, float3 domainMin, float3 domainMax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numMarbles) return;
    
    Marble* marble = &marbles[i];
    float restitution = 0.1f;  // Very low bounce
    float friction = 0.95f;    // High friction
    
    // Better floor collision with position correction
    if (marble->pos.y - marble->radius < domainMin.y) {
        marble->pos.y = domainMin.y + marble->radius + 0.001f; // Small buffer
        if (marble->vel.y < 0) {
            marble->vel.y = -marble->vel.y * restitution;
            marble->vel.x *= friction;
            marble->vel.z *= friction;
            marble->angular_vel *= friction;
        }
    }
    
    // Similar fixes for other boundaries...
    if (marble->pos.y + marble->radius > domainMax.y) {
        marble->pos.y = domainMax.y - marble->radius - 0.001f;
        if (marble->vel.y > 0) {
            marble->vel.y = -marble->vel.y * restitution;
            marble->vel.x *= friction;
            marble->vel.z *= friction;
            marble->angular_vel *= friction;
        }
    }
    
    // X boundaries
    if (marble->pos.x - marble->radius < domainMin.x) {
        marble->pos.x = domainMin.x + marble->radius + 0.001f;
        if (marble->vel.x < 0) {
            marble->vel.x = -marble->vel.x * restitution;
            marble->vel.y *= friction;
            marble->vel.z *= friction;
            marble->angular_vel *= friction;
        }
    }
    
    if (marble->pos.x + marble->radius > domainMax.x) {
        marble->pos.x = domainMax.x - marble->radius - 0.001f;
        if (marble->vel.x > 0) {
            marble->vel.x = -marble->vel.x * restitution;
            marble->vel.y *= friction;
            marble->vel.z *= friction;
            marble->angular_vel *= friction;
        }
    }
    
    // Z boundaries
    if (marble->pos.z - marble->radius < domainMin.z) {
        marble->pos.z = domainMin.z + marble->radius + 0.001f;
        if (marble->vel.z < 0) {
            marble->vel.z = -marble->vel.z * restitution;
            marble->vel.x *= friction;
            marble->vel.y *= friction;
            marble->angular_vel *= friction;
        }
    }
    
    if (marble->pos.z + marble->radius > domainMax.z) {
        marble->pos.z = domainMax.z - marble->radius - 0.001f;
        if (marble->vel.z > 0) {
            marble->vel.z = -marble->vel.z * restitution;
            marble->vel.x *= friction;
            marble->vel.y *= friction;
            marble->angular_vel *= friction;
        }
    }
}

__global__ void applyEnergyDissipation(Marble* marbles, int numMarbles, float dampingFactor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numMarbles) return;
    
    // Adaptive damping based on velocity
    float linear_damping = 0.998f;
    float angular_damping = 0.995f;
    
    float speed = length(marbles[i].vel);
    if (speed > 10.0f) {
        linear_damping = 0.99f; // More damping for fast marbles
    }
    
    float ang_speed = length(marbles[i].angular_vel);
    if (ang_speed > 5.0f) {
        angular_damping = 0.98f; // More angular damping for fast rotation
    }
    
    marbles[i].vel *= linear_damping;
    marbles[i].angular_vel *= angular_damping;
}