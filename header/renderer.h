#ifndef RENDERER_H
#define RENDERER_H

#include <string>
struct float3;

// Initialization and Shutdown
void initRenderer(int Nx, int Ny);
void cleanupRenderer();

// OpenGL + GUI control
bool windowShouldClose();
void beginFrame();
void endFrame();

// CUDA–OpenGL interop
void renderCloth(int step, float3* d_clothPositions);

// GUI controls
void renderGUI(float3* forceDir,
               float* strength,
               int* tStart,
               int* tEnd,
               int* radius,
               std::string* mode);

#endif
