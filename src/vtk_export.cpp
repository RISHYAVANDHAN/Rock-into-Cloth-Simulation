#include "vtk_export.h"
#include <iostream>

void writeClothToVTK(const std::string& filename, 
                     const std::vector<ClothNode>& nodes,
                     const std::vector<Spring>& springs,
                     int num_x, int num_y) {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Error opening VTK file: " << filename << std::endl;
        return;
    }

    // Write VTK header
    out << "# vtk DataFile Version 3.0\n";
    out << "Cloth Simulation Mesh\n";
    out << "ASCII\n";
    out << "DATASET UNSTRUCTURED_GRID\n\n";

    // Write points
    out << "POINTS " << nodes.size() << " float\n";
    for (const auto& node : nodes) {
        out << node.pos.x << " " << node.pos.y << " " << node.pos.z << "\n";
    }
    out << "\n";

    // Write cells (quads for mesh)
    const int num_quads = (num_x - 1) * (num_y - 1);
    out << "CELLS " << num_quads << " " << num_quads * 5 << "\n";
    for (int y = 0; y < num_y - 1; ++y) {
        for (int x = 0; x < num_x - 1; ++x) {
            int idx0 = y * num_x + x;
            int idx1 = y * num_x + x + 1;
            int idx2 = (y + 1) * num_x + x + 1;
            int idx3 = (y + 1) * num_x + x;
            out << "4 " << idx0 << " " << idx1 << " " << idx2 << " " << idx3 << "\n";
        }
    }
    out << "\n";

    // Write cell types (quad = 9)
    out << "CELL_TYPES " << num_quads << "\n";
    for (int i = 0; i < num_quads; ++i) {
        out << "9\n";  // VTK_QUAD
    }
    out << "\n";

    // Write point data (pinned status)
    out << "POINT_DATA " << nodes.size() << "\n";
    out << "SCALARS pinned int\n";
    out << "LOOKUP_TABLE default\n";
    for (const auto& node : nodes) {
        out << (node.pinned ? 1 : 0) << "\n";
    }
    out << "\n";

    // Write spring data (optional)
    out << "CELL_DATA " << springs.size() << "\n";
    out << "SCALARS spring_type int\n";
    out << "LOOKUP_TABLE default\n";
    for (const auto& spring : springs) {
        // Classify spring types:
        // 1 = horizontal, 2 = vertical, 3 = diagonal
        int type = 3;  // diagonal by default
        if (abs(spring.j - spring.i) == 1) type = 1;  // horizontal
        else if (abs(spring.j - spring.i) == num_x) type = 2;  // vertical
        out << type << "\n";
    }

    out.close();
    std::cout << "Saved cloth mesh to: " << filename << std::endl;
}