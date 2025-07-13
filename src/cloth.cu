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

__global__ void applyClothSelfContact(ClothNode* nodes,const int* cellHead,const int* nodeNext,int3 gridSize,float3 gridMin,float cellSize) {
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
                int hash = computeHash(neighbor_cell, gridSize);
                
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


// FIXED: Proper bidirectional cloth-marble interaction
__global__ void computeMarbleClothInteraction(ClothNode* nodes,Marble* marbles,int numMarbles,float dt) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_nodes = num_x * num_y;
    if (node_idx >= total_nodes) return;

    ClothNode* node = &nodes[node_idx];
    if (node->pinned) return;

    for (int i = 0; i < numMarbles; i++) {
        Marble* marble = &marbles[i];
        float3 delta = node->pos - marble->pos;
        float dist = length(delta);
        
        // FIXED: Improved collision detection threshold
        float collision_threshold = marble->radius + 0.01f; // Small buffer for stability
        
        if (dist < collision_threshold && dist > 1e-8f) {
            float3 normal = delta / dist;
            float penetration = collision_threshold - dist;
            
            // FIXED: Better contact point calculation
            float3 contact_point = marble->pos + normal * marble->radius;
            float3 r_marble = contact_point - marble->pos;
            
            // Relative velocity calculation
            float3 v_marble_contact = marble->vel + cross(marble->angular_vel, r_marble);
            float3 v_rel = node->vel - v_marble_contact;
            float vn = dot(v_rel, normal);
            
            // FIXED: Proper reduced mass calculation
            float reduced_mass = (node_mass * marble->mass) / (node_mass + marble->mass);
            float kd_critical = 2.0f * sqrtf(reduced_mass * kn_marble) * 0.3f; // Reduced damping
            
            // FIXED: Stronger spring force for better separation
            float3 Fn = kn_marble * penetration * normal - kd_critical * vn * normal;
            
            // FIXED: Improved friction model
            float3 v_tangent = v_rel - vn * normal;
            float vt_mag = length(v_tangent);
            float3 Ft = make_float3(0,0,0);
            
            if (vt_mag > 1e-6f) {
                float3 tangent = v_tangent / vt_mag;
                float max_ft = friction_mu_marble * length(Fn);
                Ft = -fminf(max_ft, kd_critical * vt_mag) * tangent;
            }
            
            float3 F_total = Fn + Ft;
            
            // FIXED: Apply forces with proper scaling
            float force_scale = 1.0f;
            if (penetration > marble->radius * 0.5f) {
                // Increase force for deep penetration
                force_scale = 1.0f + (penetration / marble->radius);
            }
            
            F_total *= force_scale;
            
            // Apply to cloth node
            atomicAdd(&node->force.x, F_total.x);
            atomicAdd(&node->force.y, F_total.y);
            atomicAdd(&node->force.z, F_total.z);
            
            // FIXED: Apply reaction force to marble
            atomicAdd(&marble->force.x, -F_total.x);
            atomicAdd(&marble->force.y, -F_total.y);
            atomicAdd(&marble->force.z, -F_total.z);
            
            // FIXED: Apply torque to marble
            float3 tau = cross(r_marble, -F_total);
            atomicAdd(&marble->torque.x, tau.x);
            atomicAdd(&marble->torque.y, tau.y);
            atomicAdd(&marble->torque.z, tau.z);
        }
    }
}


__device__ void checkSphereTriangleCollision(Marble* marble,ClothNode* v0,ClothNode* v1,ClothNode* v2) {
    // Calculate triangle normal
    float3 edge1 = v1->pos - v0->pos;
    float3 edge2 = v2->pos - v0->pos;
    float3 normal = normalize(cross(edge1, edge2));
    
    // Distance from sphere center to triangle plane
    float3 to_sphere = marble->pos - v0->pos;
    float dist_to_plane = dot(to_sphere, normal);
    
    // Check if sphere is close enough to triangle
    if (fabsf(dist_to_plane) < marble->radius) {
        // Find closest point on triangle
        float3 closest_point = marble->pos - dist_to_plane * normal;
        
        // Check if closest point is inside triangle (barycentric coordinates)
        float3 v0_to_closest = closest_point - v0->pos;
        float d00 = dot(edge1, edge1);
        float d01 = dot(edge1, edge2);
        float d11 = dot(edge2, edge2);
        float d20 = dot(v0_to_closest, edge1);
        float d21 = dot(v0_to_closest, edge2);
        
        float denom = d00 * d11 - d01 * d01;
        if (fabsf(denom) < 1e-6f) return;
        
        float v = (d11 * d20 - d01 * d21) / denom;
        float w = (d00 * d21 - d01 * d20) / denom;
        float u = 1.0f - v - w;
        
        // Check if inside triangle
        if (u >= 0 && v >= 0 && w >= 0) {
            float penetration = marble->radius - fabsf(dist_to_plane);
            if (penetration > 0) {
                // Calculate contact force
                float3 contact_normal = (dist_to_plane < 0) ? -normal : normal;
                
                // Relative velocity
                float3 marble_vel_at_contact = marble->vel;
                float3 triangle_vel = u * v0->vel + v * v1->vel + w * v2->vel;
                float3 v_rel = marble_vel_at_contact - triangle_vel;
                float vn = dot(v_rel, contact_normal);
                
                // Contact force
                float3 Fn = kn_marble * penetration * contact_normal - kd_marble * vn * contact_normal;
                
                // Apply forces to marble
                atomicAdd(&marble->force.x, -Fn.x);
                atomicAdd(&marble->force.y, -Fn.y);
                atomicAdd(&marble->force.z, -Fn.z);
                
                // Distribute forces to triangle vertices
                float3 force_per_vertex = Fn / 3.0f;
                if (!v0->pinned) {
                    atomicAdd(&v0->force.x, u * force_per_vertex.x);
                    atomicAdd(&v0->force.y, u * force_per_vertex.y);
                    atomicAdd(&v0->force.z, u * force_per_vertex.z);
                }
                if (!v1->pinned) {
                    atomicAdd(&v1->force.x, v * force_per_vertex.x);
                    atomicAdd(&v1->force.y, v * force_per_vertex.y);
                    atomicAdd(&v1->force.z, v * force_per_vertex.z);
                }
                if (!v2->pinned) {
                    atomicAdd(&v2->force.x, w * force_per_vertex.x);
                    atomicAdd(&v2->force.y, w * force_per_vertex.y);
                    atomicAdd(&v2->force.z, w * force_per_vertex.z);
                }
            }
        }
    }
}

__global__ void computeMarbleTriangleCollision(ClothNode* nodes,Marble* marbles,int numMarbles) {
    int marble_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (marble_idx >= numMarbles) return;
    
    Marble* marble = &marbles[marble_idx];
    
    // Check collision with cloth triangles
    for (int y = 0; y < num_y - 1; y++) {
        for (int x = 0; x < num_x - 1; x++) {
            // Two triangles per quad
            int idx1 = y * num_x + x;
            int idx2 = y * num_x + (x + 1);
            int idx3 = (y + 1) * num_x + x;
            int idx4 = (y + 1) * num_x + (x + 1);
            
            // Triangle 1: (idx1, idx2, idx3)
            checkSphereTriangleCollision(marble, &nodes[idx1], &nodes[idx2], &nodes[idx3]);
            
            // Triangle 2: (idx2, idx3, idx4)
            checkSphereTriangleCollision(marble, &nodes[idx2], &nodes[idx3], &nodes[idx4]);
        }
    }
}


__global__ void applyMarbleMarbleContact(Marble* marbles, int numMarbles,const int* cellHead, const int* marbleNext,int3 gridSize, float3 gridMin, float cellSize) {
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
                int hash = computeHash(neighborPos, gridSize);
                int j = cellHead[hash];

                while (j != -1) {
                    if (j > i) {
                        Marble& mj = marbles[j];
                        float3 delta = mj.pos - mi.pos;
                        float dist2 = dot(delta, delta);

                        if (dist2 < radiusSq && dist2 > 1e-12f) {
                            float dist = sqrtf(dist2);
                            float3 normal = delta / dist;
                            float overlap = contactRadius2 - dist;

                            float3 r_ij = mj.pos - mi.pos;
                            float3 v_rel = mj.vel - mi.vel;
                            float vn = dot(v_rel, normal);

                            float3 Fn = kn_contact * overlap * normal - kd_contact * vn * normal;

                            float3 v_t = v_rel - vn * normal;
                            float vt_mag = length(v_t);
                            float3 Ft = make_float3(0.0, 0.0, 0.0);
                            if (vt_mag > 1e-6f) {
                                float3 tangent = v_t / vt_mag;
                                float maxFt = friction_mu * length(Fn);
                                Ft = -min(maxFt, kd_contact * vt_mag) * tangent;
                            }

                            float3 F_total = Fn + Ft;

                            atomicAdd(&mi.force.x, -F_total.x);
                            atomicAdd(&mi.force.y, -F_total.y);
                            atomicAdd(&mi.force.z, -F_total.z);

                            atomicAdd(&mj.force.x, F_total.x);
                            atomicAdd(&mj.force.y, F_total.y);
                            atomicAdd(&mj.force.z, F_total.z);

                            float3 tau_i = cross(-r_ij, F_total);
                            float3 tau_j = cross(r_ij, -F_total);

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
