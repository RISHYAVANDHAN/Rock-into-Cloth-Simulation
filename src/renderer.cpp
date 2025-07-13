#include "renderer.h"
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
#include <vector>
#include <cmath>

static GLFWwindow* window = nullptr;
static GLuint vbo = 0;
static GLuint vao = 0;
static GLuint ebo = 0;
static GLuint shaderProgram = 0;
static cudaGraphicsResource* cuda_vbo_res = nullptr;
static int numVertices = 0;
static int cloth_nx = 0, cloth_ny = 0;
static bool cudaInteropEnabled = false;
static bool wireframe = false;
static int numIndices = 0;

// Marble rendering data
static GLuint marbleVBO = 0;
static GLuint marbleVAO = 0;
static GLuint marbleEBO = 0;
static GLuint marbleShaderProgram = 0;
static std::vector<float> sphereVertices;
static std::vector<unsigned int> sphereIndices;
static int sphereVertexCount = 0;
static int sphereIndexCount = 0;

// Host buffer for fallback rendering
static float* h_vertex_buffer = nullptr;
static float* h_marble_buffer = nullptr;
static int maxMarbles = 0;

// Rendering options
static bool showMarbles = true;
static bool showCloth = true;
static float marbleAlpha = 1.0f;

// Camera variables - positioned to see falling cloth from diagonal angle
static glm::vec3 cameraPos = glm::vec3(3.0f, 2.5f, 3.0f);
static glm::vec3 cameraFront = glm::vec3(-0.5f, -0.3f, -0.5f);
static glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
static float yaw = -135.0f;
static float pitch = -20.0f;
static float lastX = 640.0f;
static float lastY = 360.0f;
static bool firstMouse = true;
static float fov = 60.0f;

// Function declarations
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
GLuint compileShaders();
GLuint compileMarbleShaders();
void checkCudaError(cudaError_t error, const char* operation);
void createClothMesh();
void createSphereMesh();
void initMarbleRenderer(int maxMarbleCount);
void renderMarbles(const std::vector<Marble>& marbles, const glm::mat4& view, const glm::mat4& projection);

void checkCudaError(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << operation << " failed: " << cudaGetErrorString(error) << std::endl;
    }
}

void createSphereMesh() {
    sphereVertices.clear();
    sphereIndices.clear();
    
    const int sectors = 20;
    const int stacks = 20;
    const float radius = 1.0f;
    
    // Generate vertices
    for (int i = 0; i <= stacks; ++i) {
        float stackAngle = M_PI / 2 - i * M_PI / stacks;
        float xy = radius * cosf(stackAngle);
        float z = radius * sinf(stackAngle);
        
        for (int j = 0; j <= sectors; ++j) {
            float sectorAngle = j * 2 * M_PI / sectors;
            float x = xy * cosf(sectorAngle);
            float y = xy * sinf(sectorAngle);
            
            sphereVertices.push_back(x);
            sphereVertices.push_back(y);
            sphereVertices.push_back(z);
            
            // Normal (same as position for unit sphere)
            sphereVertices.push_back(x);
            sphereVertices.push_back(y);
            sphereVertices.push_back(z);
        }
    }
    
    // Generate indices
    for (int i = 0; i < stacks; ++i) {
        int k1 = i * (sectors + 1);
        int k2 = k1 + sectors + 1;
        
        for (int j = 0; j < sectors; ++j, ++k1, ++k2) {
            if (i != 0) {
                sphereIndices.push_back(k1);
                sphereIndices.push_back(k2);
                sphereIndices.push_back(k1 + 1);
            }
            
            if (i != (stacks - 1)) {
                sphereIndices.push_back(k1 + 1);
                sphereIndices.push_back(k2);
                sphereIndices.push_back(k2 + 1);
            }
        }
    }
    
    sphereVertexCount = sphereVertices.size() / 6; // 3 pos + 3 normal
    sphereIndexCount = sphereIndices.size();
}

void createClothMesh() {
    // Create indices for triangles
    std::vector<unsigned int> indices;
    
    for (int y = 0; y < cloth_ny - 1; y++) {
        for (int x = 0; x < cloth_nx - 1; x++) {
            int topLeft = y * cloth_nx + x;
            int topRight = y * cloth_nx + (x + 1);
            int bottomLeft = (y + 1) * cloth_nx + x;
            int bottomRight = (y + 1) * cloth_nx + (x + 1);
            
            // First triangle (counter-clockwise)
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);
            
            // Second triangle (counter-clockwise)
            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }
    
    numIndices = indices.size();
    
    // Create and fill element buffer
    glGenBuffers(1, &ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), 
                 indices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void initMarbleRenderer(int maxMarbleCount) {
    maxMarbles = maxMarbleCount;
    
    // Create sphere geometry
    createSphereMesh();
    
    // Create marble VAO
    glGenVertexArrays(1, &marbleVAO);
    glBindVertexArray(marbleVAO);
    
    // Create marble VBO
    glGenBuffers(1, &marbleVBO);
    glBindBuffer(GL_ARRAY_BUFFER, marbleVBO);
    glBufferData(GL_ARRAY_BUFFER, sphereVertices.size() * sizeof(float), 
                 sphereVertices.data(), GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Create marble EBO
    glGenBuffers(1, &marbleEBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, marbleEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphereIndices.size() * sizeof(unsigned int),
                 sphereIndices.data(), GL_STATIC_DRAW);
    
    glBindVertexArray(0);
    
    // Create marble shader program
    marbleShaderProgram = compileMarbleShaders();
    if (!marbleShaderProgram) {
        std::cerr << "Failed to compile marble shaders" << std::endl;
    }
    
    // Allocate host buffer for marble data
    h_marble_buffer = new float[maxMarbles * 4]; // x, y, z, radius
    if (!h_marble_buffer) {
        std::cerr << "Failed to allocate host marble buffer" << std::endl;
    }
}

void initRenderer(int Nx, int Ny, int maxMarbleCount) {
    numVertices = Nx * Ny;
    cloth_nx = Nx;
    cloth_ny = Ny;

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1280, 720, "Cloth Simulation with Marbles", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // Set callbacks
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return;
    }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "GPU: " << glGetString(GL_RENDERER) << std::endl;

    // Initialize CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    checkCudaError(cudaStatus, "cudaSetDevice");

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "CUDA Device: " << deviceProp.name << std::endl;

    // Setup ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Create cloth shaders
    shaderProgram = compileShaders();
    if (!shaderProgram) {
        std::cerr << "Failed to compile cloth shaders" << std::endl;
        return;
    }

    // Create cloth VAO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Create cloth VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, numVertices * sizeof(float3), nullptr, GL_DYNAMIC_DRAW);
    
    // Set vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);
    
    // Create cloth mesh
    createClothMesh();
    
    // Bind element buffer to VAO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // Initialize marble renderer
    initMarbleRenderer(maxMarbleCount);

    // Try CUDA-OpenGL interop
    glfwMakeContextCurrent(window);
    glFinish();
    
    cudaStatus = cudaGraphicsGLRegisterBuffer(&cuda_vbo_res, vbo, cudaGraphicsMapFlagsWriteDiscard);
    
    if (cudaStatus == cudaSuccess) {
        cudaInteropEnabled = true;
        std::cout << "CUDA-OpenGL interop enabled successfully" << std::endl;
    } else {
        cudaInteropEnabled = false;
        std::cerr << "CUDA-OpenGL interop failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        std::cerr << "Falling back to CPU-based rendering" << std::endl;
        
        h_vertex_buffer = new float[numVertices * 3];
        if (!h_vertex_buffer) {
            std::cerr << "Failed to allocate host vertex buffer" << std::endl;
        }
    }

    // OpenGL setup
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_CULL_FACE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0f, 1.0f);
}

GLuint compileShaders() {
    // Vertex shader with basic lighting and proper normals
    const char* vertexShaderSource = R"(
        #version 330 core
        layout(location = 0) in vec3 position;
        
        uniform mat4 MVP;
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec3 WorldPos;
        
        void main() {
            WorldPos = vec3(model * vec4(position, 1.0));
            FragPos = WorldPos;
            Normal = normalize(vec3(0.0, 0.0, 1.0));
            gl_Position = MVP * vec4(position, 1.0);
        }
    )";
    
    // Fragment shader with better lighting and color
    const char* fragmentShaderSource = R"(
        #version 330 core
        in vec3 FragPos;
        in vec3 Normal;
        in vec3 WorldPos;
        
        out vec4 FragColor;
        
        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec3 lightColor;
        uniform vec3 objectColor;
        uniform bool wireframeMode;
        
        void main() {
            if (wireframeMode) {
                FragColor = vec4(1.0, 1.0, 1.0, 1.0);
                return;
            }
            
            float ambientStrength = 0.4;
            vec3 ambient = ambientStrength * lightColor;
            
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            
            float specularStrength = 0.3;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16);
            vec3 specular = specularStrength * spec * lightColor;
            
            float variation = sin(WorldPos.x * 3.0) * sin(WorldPos.y * 3.0) * 0.1 + 1.0;
            
            vec3 result = (ambient + diffuse + specular) * objectColor * variation;
            FragColor = vec4(result, 1.0);
        }
    )";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
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
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Fragment shader compilation failed:\n" << infoLog << std::endl;
        return 0;
    }
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
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

GLuint compileMarbleShaders() {
    const char* vertexShaderSource = R"(
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        uniform vec3 marblePos;
        uniform float marbleRadius;
        
        out vec3 FragPos;
        out vec3 Normal;
        out vec3 WorldPos;
        
        void main() {
            vec3 scaledPos = position * marbleRadius + marblePos;
            WorldPos = vec3(model * vec4(scaledPos, 1.0));
            FragPos = WorldPos;
            Normal = normalize(mat3(transpose(inverse(model))) * normal);
            gl_Position = projection * view * model * vec4(scaledPos, 1.0);
        }
    )";
    
    const char* fragmentShaderSource = R"(
        #version 330 core
        in vec3 FragPos;
        in vec3 Normal;
        in vec3 WorldPos;
        
        out vec4 FragColor;
        
        uniform vec3 lightPos;
        uniform vec3 viewPos;
        uniform vec3 lightColor;
        uniform vec3 marbleColor;
        uniform float alpha;
        
        void main() {
            // Ambient
            float ambientStrength = 0.3;
            vec3 ambient = ambientStrength * lightColor;
            
            // Diffuse
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            
            // Specular
            float specularStrength = 0.8;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
            vec3 specular = specularStrength * spec * lightColor;
            
            vec3 result = (ambient + diffuse + specular) * marbleColor;
            FragColor = vec4(result, alpha);
        }
    )";

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    GLint success;
    GLchar infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Marble vertex shader compilation failed:\n" << infoLog << std::endl;
        return 0;
    }
    
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Marble fragment shader compilation failed:\n" << infoLog << std::endl;
        return 0;
    }
    
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(program, 512, NULL, infoLog);
        std::cerr << "Marble shader linking failed:\n" << infoLog << std::endl;
        return 0;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return program;
}

void renderMarbles(const std::vector<Marble>& marbles, const glm::mat4& view, const glm::mat4& projection) {
    if (!showMarbles || !marbleShaderProgram || marbles.empty()) return;
    
    glUseProgram(marbleShaderProgram);
    
    glm::mat4 model = glm::mat4(1.0f);
    
    // Set uniforms
    GLint modelLoc = glGetUniformLocation(marbleShaderProgram, "model");
    GLint viewLoc = glGetUniformLocation(marbleShaderProgram, "view");
    GLint projLoc = glGetUniformLocation(marbleShaderProgram, "projection");
    GLint marblePosLoc = glGetUniformLocation(marbleShaderProgram, "marblePos");
    GLint marbleRadiusLoc = glGetUniformLocation(marbleShaderProgram, "marbleRadius");
    GLint lightPosLoc = glGetUniformLocation(marbleShaderProgram, "lightPos");
    GLint viewPosLoc = glGetUniformLocation(marbleShaderProgram, "viewPos");
    GLint lightColorLoc = glGetUniformLocation(marbleShaderProgram, "lightColor");
    GLint marbleColorLoc = glGetUniformLocation(marbleShaderProgram, "marbleColor");
    GLint alphaLoc = glGetUniformLocation(marbleShaderProgram, "alpha");
    
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
    
    // Lighting setup
    glm::vec3 lightPos = glm::vec3(2.0f, 4.0f, 2.0f);
    glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
    
    glUniform3fv(lightPosLoc, 1, glm::value_ptr(lightPos));
    glUniform3fv(viewPosLoc, 1, glm::value_ptr(cameraPos));
    glUniform3fv(lightColorLoc, 1, glm::value_ptr(lightColor));
    glUniform1f(alphaLoc, marbleAlpha);
    
    glBindVertexArray(marbleVAO);
    
    // Render each marble
    for (size_t i = 0; i < marbles.size(); ++i) {
        const Marble& marble = marbles[i];
        
        // Set marble-specific uniforms
        glm::vec3 marblePos = glm::vec3(marble.pos.x, marble.pos.y, marble.pos.z);
        glUniform3fv(marblePosLoc, 1, glm::value_ptr(marblePos));
        glUniform1f(marbleRadiusLoc, marble.radius);
        
        // Color based on marble properties (size, speed, etc.)
        float speedFactor = glm::length(glm::vec3(marble.vel.x, marble.vel.y, marble.vel.z)) / 10.0f;
        glm::vec3 marbleColor = glm::vec3(
            0.8f + 0.2f * sin(marble.radius * 10.0f),
            0.6f + 0.4f * speedFactor,
            0.9f - 0.3f * speedFactor
        );
        glUniform3fv(marbleColorLoc, 1, glm::value_ptr(marbleColor));
        
        // Draw marble
        glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_INT, 0);
    }
    
    glBindVertexArray(0);
}

void cleanupRenderer() {
    if (cuda_vbo_res && cudaInteropEnabled) {
        cudaGraphicsUnregisterResource(cuda_vbo_res);
        cuda_vbo_res = nullptr;
    }
    
    if (h_vertex_buffer) {
        delete[] h_vertex_buffer;
        h_vertex_buffer = nullptr;
    }
    
    if (h_marble_buffer) {
        delete[] h_marble_buffer;
        h_marble_buffer = nullptr;
    }
    
    if (vbo) {
        glDeleteBuffers(1, &vbo);
        vbo = 0;
    }
    
    if (ebo) {
        glDeleteBuffers(1, &ebo);
        ebo = 0;
    }
    
    if (vao) {
        glDeleteVertexArrays(1, &vao);
        vao = 0;
    }
    
    if (marbleVBO) {
        glDeleteBuffers(1, &marbleVBO);
        marbleVBO = 0;
    }
    
    if (marbleEBO) {
        glDeleteBuffers(1, &marbleEBO);
        marbleEBO = 0;
    }
    
    if (marbleVAO) {
        glDeleteVertexArrays(1, &marbleVAO);
        marbleVAO = 0;
    }
    
    if (shaderProgram) {
        glDeleteProgram(shaderProgram);
        shaderProgram = 0;
    }
    
    if (marbleShaderProgram) {
        glDeleteProgram(marbleShaderProgram);
        marbleShaderProgram = 0;
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
    
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void endFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
}

void renderClothAndMarbles(int step, float3* d_clothPositions, Marble* d_marbles, int numMarbles) {
    // Update camera vectors
    cameraFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront.y = sin(glm::radians(pitch));
    cameraFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(cameraFront);
    
    // Create view and projection matrices
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 projection = glm::perspective(glm::radians(fov), 1280.0f/720.0f, 0.1f, 100.0f);
    glm::mat4 model = glm::mat4(1.0f);
    
    // Render cloth
    if (showCloth && shaderProgram) {
        glm::mat4 mvp = projection * view * model;
        
        glUseProgram(shaderProgram);
        
        // Set uniforms
        GLint mvpLoc = glGetUniformLocation(shaderProgram, "MVP");
        GLint modelLoc = glGetUniformLocation(shaderProgram, "model");
        GLint viewLoc = glGetUniformLocation(shaderProgram, "view");
        GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
        GLint lightPosLoc = glGetUniformLocation(shaderProgram, "lightPos");
        GLint viewPosLoc = glGetUniformLocation(shaderProgram, "viewPos");
        GLint lightColorLoc = glGetUniformLocation(shaderProgram, "lightColor");
        GLint objectColorLoc = glGetUniformLocation(shaderProgram, "objectColor");
        GLint wireframeLoc = glGetUniformLocation(shaderProgram, "wireframeMode");
        
        glUniformMatrix4fv(mvpLoc, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));
        
        // Lighting setup
        glm::vec3 lightPos = glm::vec3(2.0f, 4.0f, 2.0f);
        glm::vec3 lightColor = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 objectColor = glm::vec3(0.8f, 0.3f, 0.3f);  // Red cloth
        
        glUniform3fv(lightPosLoc, 1, glm::value_ptr(lightPos));
        glUniform3fv(viewPosLoc, 1, glm::value_ptr(cameraPos));
        glUniform3fv(lightColorLoc, 1, glm::value_ptr(lightColor));
        glUniform3fv(objectColorLoc, 1, glm::value_ptr(objectColor));
        glUniform1i(wireframeLoc, wireframe ? 1 : 0);
        
        // Update vertex data
        bool dataUpdated = false;
        if (cudaInteropEnabled && cuda_vbo_res) {
            cudaError_t cudaStatus = cudaGraphicsMapResources(1, &cuda_vbo_res, 0);
            if (cudaStatus == cudaSuccess) {
                float3* dptr = nullptr;
                size_t size = 0;
                cudaStatus = cudaGraphicsResourceGetMappedPointer((void**)&dptr, &size, cuda_vbo_res);
                if (cudaStatus == cudaSuccess) {
                    cudaStatus = cudaMemcpy(dptr, d_clothPositions, numVertices * sizeof(float3), cudaMemcpyDeviceToDevice);
                    if (cudaStatus == cudaSuccess) {
                        dataUpdated = true;
                    }
                }
                cudaGraphicsUnmapResources(1, &cuda_vbo_res, 0);
            }
        }
        
        if (!dataUpdated && h_vertex_buffer) {
            cudaError_t cudaStatus = cudaMemcpy(h_vertex_buffer, d_clothPositions, 
                                               numVertices * sizeof(float3), cudaMemcpyDeviceToHost);
            if (cudaStatus == cudaSuccess) {
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferSubData(GL_ARRAY_BUFFER, 0, numVertices * sizeof(float3), h_vertex_buffer);
                glBindBuffer(GL_ARRAY_BUFFER, 0);
                dataUpdated = true;
            }
        }

        // Render cloth
        glBindVertexArray(vao);
        
        if (wireframe) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glLineWidth(1.0f);
        } else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
        
        glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);  // Reset
    }
    
    // Render marbles
    if (showMarbles && numMarbles > 0 && marbleShaderProgram) {
        // Copy marble data from device to host
        std::vector<Marble> marbles(numMarbles);
        cudaMemcpy(marbles.data(), d_marbles, numMarbles * sizeof(Marble), cudaMemcpyDeviceToHost);
        
        // Render marbles with the same view and projection
        renderMarbles(marbles, view, projection);
    }
    
    // UI
    static float fps = 0.0f;
    static float lastTime = 0.0f;
    float currentTime = glfwGetTime();
    float delta = currentTime - lastTime;
    
    if (delta >= 1.0f) {
        fps = ImGui::GetIO().Framerate;
        lastTime = currentTime;
    }
    
    ImGui::Begin("Cloth Simulation Control");
    ImGui::Text("FPS: %.1f", fps);
    ImGui::Text("Step: %d", step);
    ImGui::Text("Vertices: %d", numVertices);
    ImGui::Text("Triangles: %d", numIndices / 3);
    ImGui::Text("Cloth Size: %dx%d", cloth_nx, cloth_ny);
    ImGui::Text("Marbles: %d", numMarbles);
    ImGui::Text("CUDA Interop: %s", cudaInteropEnabled ? "Enabled" : "Disabled");
    
    ImGui::Separator();
    ImGui::Checkbox("Wireframe", &wireframe);
    ImGui::Checkbox("Show Cloth", &showCloth);
    ImGui::Checkbox("Show Marbles", &showMarbles);
    ImGui::SliderFloat("Marble Alpha", &marbleAlpha, 0.1f, 1.0f);
    
    if (ImGui::Button("Reset Camera")) {
        cameraPos = glm::vec3(3.0f, 2.5f, 3.0f);
        yaw = -135.0f;
        pitch = -20.0f;
        fov = 60.0f;
    }
    
    ImGui::Separator();
    ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", cameraPos.x, cameraPos.y, cameraPos.z);
    ImGui::Text("Yaw: %.1f°, Pitch: %.1f°", yaw, pitch);
    ImGui::Text("FOV: %.1f°", fov);
    
    ImGui::Separator();
    ImGui::Text("Controls:");
    ImGui::Text("Left Mouse + Drag: Rotate camera");
    ImGui::Text("Mouse Wheel: Zoom");
    ImGui::Text("W: Toggle wireframe");
    ImGui::Text("R: Reset camera");
    ImGui::Text("ESC: Exit");
    
    ImGui::End();
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
        if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
        }

        float xoffset = xpos - lastX;
        float yoffset = lastY - ypos;
        lastX = xpos;
        lastY = ypos;

        float sensitivity = 0.1f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        // Constrain pitch
        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;
    } else {
        firstMouse = true;
    }
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    fov -= (float)yoffset * 2.0f;
    if (fov < 10.0f) fov = 10.0f;
    if (fov > 120.0f) fov = 120.0f;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_W:
                wireframe = !wireframe;
                break;
            case GLFW_KEY_R:
                // Reset camera to good viewing angle
                cameraPos = glm::vec3(3.0f, 2.5f, 3.0f);
                yaw = -135.0f;
                pitch = -20.0f;
                fov = 60.0f;
                break;
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
        }
    }
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}