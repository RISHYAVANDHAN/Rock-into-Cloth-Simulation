// ==================== cloth.cu ====================
#include "cloth.cuh"
#include "spatial_hash.cuh"
#include "params.cuh"
#include "marble.cuh"
#include <math.h>

__host__ void initializeClothGrid(ClothNode* d_nodes, int num_x, int num_y, float clothWidth, float clothHeight) {
    ClothNode* h_nodes = new ClothNode[num_x * num_y];

    const float halfWidth = clothWidth / 2.0f;
    const float halfHeight = clothHeight / 2.0f;

    for (int y = 0; y < num_y; ++y) {
        for (int x = 0; x < num_x; ++x) {
            int i = y * num_x + x;

            float rand_z = 0.001f * ((rand() % 100) / 100.0f - 0.5f);  // Z-perturbation
            h_nodes[i].index = i;
            h_nodes[i].pos = make_float3(
                x * (clothWidth / (num_x - 1)) - halfWidth,
                rand_z,
                y * (clothHeight / (num_y - 1)) - halfHeight
            );

            h_nodes[i].vel = make_float3(0.0f, 0.0f, 0.0f);
            h_nodes[i].force = make_float3(0.0f, 0.0f, 0.0f);

            h_nodes[i].pinned =
                (x == 0 && y == 0) ||
                (x == num_x - 1 && y == 0) ||
                (x == 0 && y == num_y - 1) ||
                (x == num_x - 1 && y == num_y - 1);
        }
    }

    cudaMemcpy(d_nodes, h_nodes, num_x * num_y * sizeof(ClothNode), cudaMemcpyHostToDevice);
    delete[] h_nodes;
}

__global__ void applyGravity(ClothNode* nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (i < total && !nodes[i].pinned) {
        nodes[i].force += node_mass * gravity;
    }
}

__global__ void applyViscousDamping(ClothNode* nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (i < total && !nodes[i].pinned) {
        float speed = length(nodes[i].vel);
        nodes[i].force += -kd * speed * nodes[i].vel;
    }
}

__global__ void resetClothForces(ClothNode* nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (i < total) {
        nodes[i].force = make_float3(0.f, 0.f, 0.f);
    }
}

__global__ void applyPinningConstraints(ClothNode* nodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (i < total && nodes[i].pinned) {
        nodes[i].vel = make_float3(0.f, 0.f, 0.f);
        nodes[i].force = make_float3(0.f, 0.f, 0.f);
    }
}

__global__ void applySpringForces(ClothNode* nodes, Spring* springs, int numSprings) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSprings) return;

    Spring spring = springs[idx];
    int i = spring.i;
    int j = spring.j;
    float L0 = spring.restLength;

    float3 xi = nodes[i].pos;
    float3 xj = nodes[j].pos;
    float3 vi = nodes[i].vel;
    float3 vj = nodes[j].vel;

    float3 dir = xj - xi;
    float dist = length(dir);
    if (dist < 1e-6f) return;

    float3 n = dir / dist;

    float ks_local = 0.0f;
    switch (spring.type) {
        case 0: ks_local = ks_structural; break;
        case 1: ks_local = ks_shear; break;
        case 2: ks_local = ks_bend; break;
    }

    float3 f_spring = -ks_local * ((dist - L0) / L0) * n;

    float3 v_rel = vj - vi;
    float v_along_spring = dot(v_rel, n);
    float3 f_damp = -kd * v_along_spring * n;

    float3 f_total = f_spring + f_damp;

    if (!nodes[i].pinned) {
        atomicAdd(&nodes[i].force.x, -f_total.x);
        atomicAdd(&nodes[i].force.y, -f_total.y);
        atomicAdd(&nodes[i].force.z, -f_total.z);
    }
    if (!nodes[j].pinned) {
        atomicAdd(&nodes[j].force.x, f_total.x);
        atomicAdd(&nodes[j].force.y, f_total.y);
        atomicAdd(&nodes[j].force.z, f_total.z);
    }
}

__global__ void applyClothSelfContact(ClothNode* nodes, const int* cellHead, const int* nodeNext, int3 gridSize, float3 gridMin, float cellSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_x * num_y;
    if (i >= total || nodes[i].pinned) return;

    float3 pos_i = nodes[i].pos;
    int3 cell_coord = make_int3(
        floorf((pos_i.x - gridMin.x) / cellSize),
        floorf((pos_i.y - gridMin.y) / cellSize),
        floorf((pos_i.z - gridMin.z) / cellSize)
    );

    const float contact_radius_sq = contact_radius * contact_radius;

    // Check 3x3x3 neighborhood
    for (int z = max(0, cell_coord.z-1); z <= min(gridSize.z-1, cell_coord.z+1); z++) {
        for (int y = max(0, cell_coord.y-1); y <= min(gridSize.y-1, cell_coord.y+1); y++) {
            for (int x = max(0, cell_coord.x-1); x <= min(gridSize.x-1, cell_coord.x+1); x++) {
                int3 neighbor_cell = make_int3(x, y, z);
                int hash = calcHash(neighbor_cell, gridSize);
                
                // Traverse linked list for this cell
                int j = cellHead[hash];
                while (j != -1) {
                    if (j > i) { // Avoid duplicate pairs
                        float3 pos_j = nodes[j].pos;
                        float3 delta = pos_j - pos_i;
                        float dist_sq = dot(delta, delta);
                        
                        if (dist_sq < contact_radius_sq && dist_sq > 1e-12f) {
                            float dist = sqrtf(dist_sq);
                            float3 dir = delta / dist;
                            float overlap = 2.0f * contact_radius - dist;
                            
                            // Relative velocity
                            float3 v_rel = nodes[j].vel - nodes[i].vel;
                            float vn = dot(v_rel, dir);
                            
                            // Normal force (damped)
                            float3 f_n = kn_contact * overlap * dir - kd_contact * vn * dir;
                            
                            // Tangential friction
                            float3 v_t = v_rel - vn * dir;
                            float vt_mag = length(v_t);
                            if (vt_mag > 1e-6f) {
                                float3 tangent = v_t / vt_mag;
                                float max_ft = friction_mu * length(f_n);
                                float3 f_t = -min(max_ft, vt_mag * kd_contact) * tangent;
                                f_n += f_t;
                            }
                            
                            // Apply forces
                            if (!nodes[i].pinned) {
                                atomicAdd(&nodes[i].force.x, -f_n.x);
                                atomicAdd(&nodes[i].force.y, -f_n.y);
                                atomicAdd(&nodes[i].force.z, -f_n.z);
                            }
                            if (!nodes[j].pinned) {
                                atomicAdd(&nodes[j].force.x, f_n.x);
                                atomicAdd(&nodes[j].force.y, f_n.y);
                                atomicAdd(&nodes[j].force.z, f_n.z);
                            }
                        }
                    }
                    j = nodeNext[j];
                }
            }
        }
    }
}

// Unified marble-cloth collision detection using spatial grid
__global__ void applyMarbleClothContact(Marble* marbles, int numMarbles, ClothNode* nodes, const int* cellHead, const int* nodeNext, int3 gridSize, float3 gridMin, float cellSize) {
    int marble_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (marble_idx >= numMarbles) return;

    Marble* marble = &marbles[marble_idx];
    float3 marble_pos = marble->pos;
    float marble_radius = marble->radius;

    // Get marble's grid cell
    int3 cell_coord = make_int3(
        floorf((marble_pos.x - gridMin.x) / cellSize),
        floorf((marble_pos.y - gridMin.y) / cellSize),
        floorf((marble_pos.z - gridMin.z) / cellSize)
    );

    // Check 3x3x3 neighborhood for cloth nodes
    for (int z = max(0, cell_coord.z-1); z <= min(gridSize.z-1, cell_coord.z+1); z++) {
        for (int y = max(0, cell_coord.y-1); y <= min(gridSize.y-1, cell_coord.y+1); y++) {
            for (int x = max(0, cell_coord.x-1); x <= min(gridSize.x-1, cell_coord.x+1); x++) {
                int3 neighbor_cell = make_int3(x, y, z);
                int hash = calcHash(neighbor_cell, gridSize);
                
                // Traverse linked list for this cell
                int node_idx = cellHead[hash];
                while (node_idx != -1) {
                    // Make sure this is a cloth node (not a marble)
                    if (node_idx < num_x * num_y) {
                        ClothNode* node = &nodes[node_idx];
                        
                        float3 delta = node->pos - marble_pos;
                        float dist = length(delta);
                        float penetration = marble_radius + contact_radius - dist;
                        
                        if (penetration > 0.0f && dist > 1e-6f) {
                            float3 normal = delta / dist;
                            float3 vrel = node->vel - marble->vel;
                            
                            // Normal force with damping
                            float vn = dot(vrel, normal);
                            float3 Fn = kn_marble * penetration * normal - kd_marble * vn * normal;
                            
                            // Friction force
                            float3 vt = vrel - vn * normal;
                            float vt_mag = length(vt);
                            if (vt_mag > 1e-6f) {
                                float3 tangent = vt / vt_mag;
                                float max_ft = friction_mu * length(Fn);
                                float3 Ft = -min(max_ft, kd_marble * vt_mag) * tangent;
                                Fn += Ft;
                            }
                            
                            // Apply forces
                            if (!node->pinned) {
                                atomicAdd(&node->force.x, Fn.x);
                                atomicAdd(&node->force.y, Fn.y);
                                atomicAdd(&node->force.z, Fn.z);
                            }
                            
                            atomicAdd(&marble->force.x, -Fn.x);
                            atomicAdd(&marble->force.y, -Fn.y);
                            atomicAdd(&marble->force.z, -Fn.z);
                        }
                    }
                    node_idx = nodeNext[node_idx];
                }
            }
        }
    }
}

__global__ void applyMarbleMarbleContact(Marble* marbles, int numMarbles, const int* cellHead, const int* marbleNext, int3 gridSize, float3 gridMin, float cellSize) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numMarbles) return;

    Marble& mi = marbles[i];
    float3 pos_i = mi.pos;
    int3 gridPos = make_int3(
        floorf((pos_i.x - gridMin.x) / cellSize),
        floorf((pos_i.y - gridMin.y) / cellSize),
        floorf((pos_i.z - gridMin.z) / cellSize)
    );

    float contactRadius = mi.radius;
    float contactRadius2 = contactRadius * 2.0f;
    float radiusSq = contactRadius2 * contactRadius2;

    for (int z = max(0, gridPos.z - 1); z <= min(gridSize.z - 1, gridPos.z + 1); z++) {
        for (int y = max(0, gridPos.y - 1); y <= min(gridSize.y - 1, gridPos.y + 1); y++) {
            for (int x = max(0, gridPos.x - 1); x <= min(gridSize.x - 1, gridPos.x + 1); x++) {
                int3 neighborPos = make_int3(x, y, z);
                int hash = calcHash(neighborPos, gridSize);
                int j = cellHead[hash];

                while (j != -1) {
                    if (j > i) { // Avoid duplicate pairs and self-interaction
                        Marble& mj = marbles[j];
                        float3 delta = mj.pos - mi.pos;
                        float dist2 = dot(delta, delta);

                        if (dist2 < radiusSq && dist2 > 1e-12f) {
                            float dist = sqrtf(dist2);
                            float3 normal = delta / dist;
                            float overlap = contactRadius2 - dist;

                            float3 v_rel = mj.vel - mi.vel;
                            float vn = dot(v_rel, normal);

                            // Normal force
                            float3 Fn = kn_contact * overlap * normal - kd_contact * vn * normal;

                            // Tangential friction
                            float3 v_t = v_rel - vn * normal;
                            float vt_mag = length(v_t);
                            float3 Ft = make_float3(0.0f, 0.0f, 0.0f);
                            if (vt_mag > 1e-6f) {
                                float3 tangent = v_t / vt_mag;
                                float maxFt = friction_mu * length(Fn);
                                Ft = -min(maxFt, kd_contact * vt_mag) * tangent;
                            }

                            float3 F_total = Fn + Ft;

                            // Apply forces
                            atomicAdd(&mi.force.x, -F_total.x);
                            atomicAdd(&mi.force.y, -F_total.y);
                            atomicAdd(&mi.force.z, -F_total.z);

                            atomicAdd(&mj.force.x, F_total.x);
                            atomicAdd(&mj.force.y, F_total.y);
                            atomicAdd(&mj.force.z, F_total.z);

                            // Apply torques for rotational dynamics
                            float3 r_i = make_float3(0.0f, 0.0f, 0.0f); // Contact point relative to center
                            float3 r_j = make_float3(0.0f, 0.0f, 0.0f);
                            
                            float3 tau_i = cross(r_i, -F_total);
                            float3 tau_j = cross(r_j, F_total);

                            atomicAdd(&mi.torque.x, tau_i.x);
                            atomicAdd(&mi.torque.y, tau_i.y);
                            atomicAdd(&mi.torque.z, tau_i.z);

                            atomicAdd(&mj.torque.x, tau_j.x);
                            atomicAdd(&mj.torque.y, tau_j.y);
                            atomicAdd(&mj.torque.z, tau_j.z);
                        }
                    }
                    j = marbleNext[j];
                }
            }
        }
    }
}

__global__ void resetClothForces(ClothNode* nodes, int numNodes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNodes) {
        nodes[i].force = make_float3(0.0f, 0.0f, 0.0f);
    }
}

__global__ void limitClothNodeVelocities(ClothNode* nodes, int numNodes, float maxVel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numNodes || nodes[i].pinned) return;
    
    float speed = length(nodes[i].vel);
    if (speed > maxVel) {
        nodes[i].vel = (maxVel / speed) * nodes[i].vel;
    }
}

// Enhanced marble-cloth collision with continuous detection
__global__ void continuousMarbleClothCollision(Marble* marbles, int numMarbles, ClothNode* nodes, 
                                              const int* cellHead, const int* nodeNext, 
                                              int3 gridSize, float3 gridMin, float cellSize, float dt) {
    int marble_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (marble_idx >= numMarbles) return;

    Marble* marble = &marbles[marble_idx];
    float3 current_pos = marble->pos;
    float3 prev_pos = marble->prevPos;
    float marble_radius = marble->radius;
    
    // Sample along movement path for continuous collision detection
    const int num_samples = 3; // Adjust for accuracy vs performance
    
    for (int sample = 0; sample < num_samples; sample++) {
        float t = (float)sample / (float)(num_samples - 1);
        float3 sample_pos = prev_pos + t * (current_pos - prev_pos);
        
        int3 cell_coord = make_int3(
            floorf((sample_pos.x - gridMin.x) / cellSize),
            floorf((sample_pos.y - gridMin.y) / cellSize),
            floorf((sample_pos.z - gridMin.z) / cellSize)
        );

        // Check 3x3x3 neighborhood
        for (int z = max(0, cell_coord.z-1); z <= min(gridSize.z-1, cell_coord.z+1); z++) {
            for (int y = max(0, cell_coord.y-1); y <= min(gridSize.y-1, cell_coord.y+1); y++) {
                for (int x = max(0, cell_coord.x-1); x <= min(gridSize.x-1, cell_coord.x+1); x++) {
                    int3 neighbor_cell = make_int3(x, y, z);
                    int hash = calcHash(neighbor_cell, gridSize);
                    
                    int node_idx = cellHead[hash];
                    while (node_idx != -1) {
                        if (node_idx < num_x * num_y) { // Ensure it's a cloth node
                            ClothNode* node = &nodes[node_idx];
                            
                            float3 delta = node->pos - sample_pos;
                            float dist = length(delta);
                            float clothThickness = 0.02f; // Treat cloth as having thickness
                            float penetration = marble_radius + clothThickness - dist;
                            
                            if (penetration > 0.0f && dist > 1e-6f) {
                                float3 normal = delta / dist;
                                float3 vrel = node->vel - marble->vel;
                                
                                // Impulse-based collision response
                                float vn = dot(vrel, normal);
                                if (vn < 0) { // Objects approaching
                                    float restitution = 0.3f;
                                    float impulse_magnitude = -(1.0f + restitution) * vn;
                                    impulse_magnitude /= (1.0f/node_mass + 1.0f/marble->mass);
                                    
                                    float3 impulse = impulse_magnitude * normal;
                                    
                                    // Apply impulse
                                    if (!node->pinned) {
                                        atomicAdd(&node->vel.x, impulse.x / node_mass);
                                        atomicAdd(&node->vel.y, impulse.y / node_mass);
                                        atomicAdd(&node->vel.z, impulse.z / node_mass);
                                    }
                                    
                                    atomicAdd(&marble->vel.x, -impulse.x / marble->mass);
                                    atomicAdd(&marble->vel.y, -impulse.y / marble->mass);
                                    atomicAdd(&marble->vel.z, -impulse.z / marble->mass);
                                    
                                    // Position correction to prevent penetration
                                    float correction_factor = 0.8f;
                                    float3 correction = correction_factor * penetration * normal;
                                    
                                    if (!node->pinned) {
                                        float total_inv_mass = 1.0f/node_mass + 1.0f/marble->mass;
                                        float3 node_correction = correction * (1.0f/node_mass) / total_inv_mass;
                                        float3 marble_correction = -correction * (1.0f/marble->mass) / total_inv_mass;
                                        
                                        atomicAdd(&node->pos.x, node_correction.x);
                                        atomicAdd(&node->pos.y, node_correction.y);
                                        atomicAdd(&node->pos.z, node_correction.z);
                                        
                                        atomicAdd(&marble->pos.x, marble_correction.x);
                                        atomicAdd(&marble->pos.y, marble_correction.y);
                                        atomicAdd(&marble->pos.z, marble_correction.z);
                                    } else {
                                        // If node is pinned, move only the marble
                                        atomicAdd(&marble->pos.x, -correction.x);
                                        atomicAdd(&marble->pos.y, -correction.y);
                                        atomicAdd(&marble->pos.z, -correction.z);
                                    }
                                }
                            }
                        }
                        node_idx = nodeNext[node_idx];
                    }
                }
            }
        }
    }
}
