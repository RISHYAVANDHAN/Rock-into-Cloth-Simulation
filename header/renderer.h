#ifndef RENDERER_H
#define RENDERER_H

#include <vector>
#include <string>
#include "marble.cuh"

// Forward declarations for GLFW
struct GLFWwindow;

struct float3;

// Initialization and Shutdown
void initRenderer(int Nx, int Ny, int maxMarbleCount);
void cleanupRenderer();

// OpenGL + GUI control
bool windowShouldClose();
void beginFrame();
void endFrame();

// Rendering function with marbles support
void renderClothAndMarbles(int step, float3* d_clothPositions, Marble* d_marbles, int numMarbles);

// Callback declarations
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
void framebufferSizeCallback(GLFWwindow* window, int width, int height);

#endif