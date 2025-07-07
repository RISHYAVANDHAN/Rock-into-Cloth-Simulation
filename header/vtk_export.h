#ifndef VTK_EXPORT_H
#define VTK_EXPORT_H

#include "cloth.cuh"
#include <vector>
#include <fstream>

void writeClothToVTK(const std::string& filename, 
                     const std::vector<ClothNode>& nodes,
                     const std::vector<Spring>& springs,
                     int num_x, int num_y);

#endif