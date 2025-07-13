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

void initializeMarbles(Marble* marbles, int numMarbles, float min_radius, float max_radius, float density) {
    srand(42);
    
    for (int i = 0; i < numMarbles; ++i) {
        float radius = min_radius + (max_radius - min_radius) * (rand() / (float)RAND_MAX);
        marbles[i].radius = radius;
        marbles[i].mass = (4.0f/3.0f) * M_PI * radius*radius*radius * density;
        marbles[i].inertia = 0.4f * marbles[i].mass * radius*radius;
        
        // FIXED: Better spatial distribution
        float spawn_radius = 0.5f; // Smaller spawn area
        float angle = (2.0f * M_PI * i) / numMarbles;
        float spawn_distance = spawn_radius * sqrtf(rand() / (float)RAND_MAX);
        
        marbles[i].pos = make_float3(
            spawn_distance * cosf(angle),
            4.0f + radius + i * 0.5f, // Higher drop with staggered heights
            spawn_distance * sinf(angle)
        );
        
        // FIXED: Slower, more controlled initial velocity
        marbles[i].vel = make_float3(
            0.1f * (rand()/(float)RAND_MAX - 0.5f),
            -1.0f - i * 0.2f,  // Slower drop with staggering
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
    
    // FIXED: Add velocity limiting during integration
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

// FIXED: Improved marble-marble collision with better separation
__global__ void computeMarbleMarbleInteraction(
    Marble* marbles, int numMarbles,
    int* cellHead, int* nodeNext,
    int3 gridSize, float3 gridMin, float cellSize
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numMarbles) return;

    float3 pos_i = marbles[i].pos;
    int3 cell_coord = make_int3(
        floorf((pos_i.x - gridMin.x) / cellSize),
        floorf((pos_i.y - gridMin.y) / cellSize),
        floorf((pos_i.z - gridMin.z) / cellSize)
    );

    for (int z = max(0, cell_coord.z-1); z <= min(gridSize.z-1, cell_coord.z+1); z++) {
        for (int y = max(0, cell_coord.y-1); y <= min(gridSize.y-1, cell_coord.y+1); y++) {
            for (int x = max(0, cell_coord.x-1); x <= min(gridSize.x-1, cell_coord.x+1); x++) {
                int3 neighbor_cell = make_int3(x, y, z);
                int hash = computeHash(neighbor_cell, gridSize);
                
                int j = cellHead[hash];
                while (j != -1) {
                    if (j > i) {
                        Marble* marble_j = &marbles[j];
                        float3 delta = marble_j->pos - pos_i;
                        float dist = length(delta);
                        float contact_dist = marbles[i].radius + marble_j->radius;
                        
                        if (dist < contact_dist && dist > 1e-6f) {
                            float3 normal = delta / dist;
                            float penetration = contact_dist - dist;
                            
                            // FIXED: Better contact point calculation
                            float3 contact_point = pos_i + normal * marbles[i].radius;
                            float3 r_i = contact_point - pos_i;
                            float3 r_j = contact_point - marble_j->pos;
                            
                            // Contact velocities
                            float3 v_i_contact = marbles[i].vel + cross(marbles[i].angular_vel, r_i);
                            float3 v_j_contact = marble_j->vel + cross(marble_j->angular_vel, r_j);
                            float3 v_rel = v_j_contact - v_i_contact;
                            float vn = dot(v_rel, normal);
                            
                            // FIXED: Proper damping calculation
                            float reduced_mass = (marbles[i].mass * marble_j->mass) / 
                                                (marbles[i].mass + marble_j->mass);
                            float damping = 2.0f * sqrtf(reduced_mass * kn_marble_marble) * 0.1f;
                            
                            // Forces
                            float3 Fn = kn_marble_marble * penetration * normal - damping * vn * normal;
                            
                            // Friction
                            float3 v_tangent = v_rel - vn * normal;
                            float vt_mag = length(v_tangent);
                            float3 Ft = make_float3(0,0,0);
                            
                            if (vt_mag > 1e-6f) {
                                float3 tangent = v_tangent / vt_mag;
                                float max_ft = friction_mu_marble_marble * length(Fn);
                                Ft = -fminf(max_ft, damping * vt_mag) * tangent;
                            }
                            
                            float3 F_total = Fn + Ft;
                            
                            // Apply forces
                            atomicAdd(&marbles[i].force.x, -F_total.x);
                            atomicAdd(&marbles[i].force.y, -F_total.y);
                            atomicAdd(&marbles[i].force.z, -F_total.z);
                            
                            atomicAdd(&marble_j->force.x, F_total.x);
                            atomicAdd(&marble_j->force.y, F_total.y);
                            atomicAdd(&marble_j->force.z, F_total.z);
                            
                            // Apply torques
                            float3 tau_i = cross(r_i, -F_total);
                            atomicAdd(&marbles[i].torque.x, tau_i.x);
                            atomicAdd(&marbles[i].torque.y, tau_i.y);
                            atomicAdd(&marbles[i].torque.z, tau_i.z);
                            
                            float3 tau_j = cross(r_j, F_total);
                            atomicAdd(&marble_j->torque.x, tau_j.x);
                            atomicAdd(&marble_j->torque.y, tau_j.y);
                            atomicAdd(&marble_j->torque.z, tau_j.z);
                        }
                    }
                    j = nodeNext[j];
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
    
    // FIXED: Better floor collision with position correction
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

__global__ void limitMarbleVelocities(Marble* marbles, int numMarbles, float maxVel, float maxAngVel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numMarbles) return;
    
    // FIXED: More aggressive velocity limiting
    float max_linear_vel = 15.0f;  // Reduced from maxVel
    float max_angular_vel = 10.0f;  // Reduced from maxAngVel
    
    float speed = length(marbles[i].vel);
    if (speed > max_linear_vel) {
        marbles[i].vel = (max_linear_vel / speed) * marbles[i].vel;
    }
    
    float angSpeed = length(marbles[i].angular_vel);
    if (angSpeed > max_angular_vel) {
        marbles[i].angular_vel = (max_angular_vel / angSpeed) * marbles[i].angular_vel;
    }
}

__global__ void applyEnergyDissipation(Marble* marbles, int numMarbles, float dampingFactor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numMarbles) return;
    
    // FIXED: Adaptive damping based on velocity
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