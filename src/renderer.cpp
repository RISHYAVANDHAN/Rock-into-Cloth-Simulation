#include "../header/renderer.h"
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cuda_runtime.h>

static GLFWwindow* window = nullptr;
static GLuint vbo = 0;
static GLuint vao = 0;
static GLuint shaderProgram = 0;
static cudaGraphicsResource* cuda_vbo_res = nullptr;
static int numVertices = 0;

// Camera variables
static glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);  // Move camera back
static glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
static glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
static float yaw = -90.0f;
static float pitch = 0.0f;
static float lastX = 640.0f;
static float lastY = 360.0f;
static bool firstMouse = true;
static float fov = 45.0f;

// Mouse state
static bool mousePressed = false;

// Function declarations
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
GLuint compileShaders();

void initRenderer(int Nx, int Ny) {
    numVertices = Nx * Ny;

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1280, 720, "Cloth Simulation", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Set callbacks
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return;
    }

    // Initialize CUDA device explicitly
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Create shaders
    shaderProgram = compileShaders();
    if (!shaderProgram) {
        std::cerr << "Failed to compile shaders" << std::endl;
    }

    // Create VAO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Create VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);
    
    // Set vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Register VBO with CUDA - ensure context is current
    glfwMakeContextCurrent(window);
    cudaStatus = cudaGraphicsGLRegisterBuffer(&cuda_vbo_res, vbo, cudaGraphicsMapFlagsWriteDiscard);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Additional error info
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        std::cerr << "CUDA devices available: " << deviceCount << std::endl;
    }

    // Basic OpenGL setup
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glPointSize(3.0f);
    glEnable(GL_DEPTH_TEST);
}

GLuint compileShaders() {
    // Vertex shader
    const char* vertexShaderSource = R"(
        #version 330 core
        layout(location = 0) in vec3 position;
        uniform mat4 MVP;
        void main() {
            gl_Position = MVP * vec4(position, 1.0);
        }
    )";
    
    // Fragment shader
    const char* fragmentShaderSource = R"(
        #version 330 core
        out vec4 color;
        void main() {
            color = vec4(1.0, 0.5, 0.2, 1.0);
        }
    )";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    // Check vertex shader compilation
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Vertex shader compilation failed:\n" << infoLog << std::endl;
        return 0;
    }
    
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    
    // Check fragment shader compilation
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Fragment shader compilation failed:\n" << infoLog << std::endl;
        return 0;
    }
    
    // Link shaders
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    // Check linking errors
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Shader linking failed:\n" << infoLog << std::endl;
        return 0;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
}

void cleanupRenderer() {
    if (cuda_vbo_res) {
        cudaGraphicsUnregisterResource(cuda_vbo_res);
        cuda_vbo_res = nullptr;
    }
    
    if (vbo) {
        glDeleteBuffers(1, &vbo);
        vbo = 0;
    }
    
    if (vao) {
        glDeleteVertexArrays(1, &vao);
        vao = 0;
    }
    
    if (shaderProgram) {
        glDeleteProgram(shaderProgram);
        shaderProgram = 0;
    }
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
}

bool windowShouldClose() {
    return glfwWindowShouldClose(window);
}

void beginFrame() {
    glfwPollEvents();
    
    // Start ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    // Clear screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void endFrame() {
    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    
    // Swap buffers
    glfwSwapBuffers(window);
}

void renderCloth(int step, float3* d_clothPositions) {
    if (!cuda_vbo_res || !shaderProgram) return;

    // Update camera
    cameraFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront.y = sin(glm::radians(pitch));
    cameraFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(cameraFront);
    
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 projection = glm::perspective(glm::radians(fov), 1280.0f/720.0f, 0.1f, 100.0f);
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::rotate(model, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));  // Rotate to view properly
    glm::mat4 mvp = projection * view * model;
    
    // Use shader and set MVP
    glUseProgram(shaderProgram);
    GLint mvpLoc = glGetUniformLocation(shaderProgram, "MVP");
    glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));
    
    // Map CUDA resource
    cudaError_t cudaStatus = cudaGraphicsMapResources(1, &cuda_vbo_res);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaGraphicsMapResources failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }

    // Get mapped pointer
    float3* dptr = nullptr;
    size_t size = 0;
    cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, cuda_vbo_res);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaGraphicsResourceGetMappedPointer failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_vbo_res);
        return;
    }

    // Copy positions to VBO
    cudaStatus = cudaMemcpy(dptr, d_clothPositions, numVertices * sizeof(float3), cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    // Unmap CUDA resource
    cudaGraphicsUnmapResources(1, &cuda_vbo_res);

    // Render points
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, numVertices);
    glBindVertexArray(0);
    
    // Render FPS counter
    static float fps = 0.0f;
    static float lastTime = 0.0f;
    float currentTime = glfwGetTime();
    float delta = currentTime - lastTime;
    
    if (delta >= 1.0f) {
        fps = ImGui::GetIO().Framerate;
        lastTime = currentTime;
    }
    
    ImGui::Begin("Performance");
    ImGui::Text("FPS: %.1f", fps);
    ImGui::End();
}

// Camera control callbacks
void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
        lastX = xpos;
        lastY = ypos;

        float sensitivity = 0.1f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        // Constrain pitch to avoid screen flip
        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;
    } else {
        firstMouse = true;
    }
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    fov -= (float)yoffset;
    if (fov < 1.0f) fov = 1.0f;
    if (fov > 90.0f) fov = 90.0f;
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}