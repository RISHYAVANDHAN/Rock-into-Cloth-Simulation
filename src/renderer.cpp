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
#include <vector>

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

// Host buffer for fallback rendering
static float* h_vertex_buffer = nullptr;

// Camera variables - positioned to see falling cloth from diagonal angle
static glm::vec3 cameraPos = glm::vec3(3.0f, 2.5f, 3.0f);  // Diagonal view to see all corners
static glm::vec3 cameraFront = glm::vec3(-0.5f, -0.3f, -0.5f);  // Looking toward origin and down
static glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
static float yaw = -135.0f;  // Diagonal angle
static float pitch = -20.0f;  // Look down to see cloth falling
static float lastX = 640.0f;
static float lastY = 360.0f;
static bool firstMouse = true;
static float fov = 60.0f;  // Wider FOV to see more

// Function declarations
void mouseCallback(GLFWwindow* window, double xpos, double ypos);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
GLuint compileShaders();
void checkCudaError(cudaError_t error, const char* operation);
void createClothMesh();

void checkCudaError(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << operation << " failed: " << cudaGetErrorString(error) << std::endl;
    }
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

void initRenderer(int Nx, int Ny) {
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

    window = glfwCreateWindow(1280, 720, "Cloth Simulation", nullptr, nullptr);
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

    // Create shaders
    shaderProgram = compileShaders();
    if (!shaderProgram) {
        std::cerr << "Failed to compile shaders" << std::endl;
        return;
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
    
    // Create cloth mesh
    createClothMesh();
    
    // Bind element buffer to VAO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

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
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f);  // Darker background for better contrast
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    
    // Disable face culling initially to see both sides of cloth
    glDisable(GL_CULL_FACE);
    
    // Enable polygon offset for better wireframe rendering
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
            
            // Simple normal calculation - assume mostly flat cloth with slight variations
            // This is a simplified approach; ideally normals should be computed from geometry
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
                FragColor = vec4(1.0, 1.0, 1.0, 1.0);  // White wireframe
                return;
            }
            
            // Ambient lighting
            float ambientStrength = 0.4;
            vec3 ambient = ambientStrength * lightColor;
            
            // Diffuse lighting
            vec3 norm = normalize(Normal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            
            // Specular lighting
            float specularStrength = 0.3;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16);
            vec3 specular = specularStrength * spec * lightColor;
            
            // Add some variation based on position for visual interest
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

void cleanupRenderer() {
    if (cuda_vbo_res && cudaInteropEnabled) {
        cudaGraphicsUnregisterResource(cuda_vbo_res);
        cuda_vbo_res = nullptr;
    }
    
    if (h_vertex_buffer) {
        delete[] h_vertex_buffer;
        h_vertex_buffer = nullptr;
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

void renderCloth(int step, float3* d_clothPositions) {
    if (!shaderProgram) return;

    // Update camera vectors
    cameraFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront.y = sin(glm::radians(pitch));
    cameraFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(cameraFront);
    
    // Create view and projection matrices
    glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    glm::mat4 projection = glm::perspective(glm::radians(fov), 1280.0f/720.0f, 0.1f, 100.0f);
    glm::mat4 model = glm::mat4(1.0f);  // No scaling - cloth is already in correct size
    glm::mat4 mvp = projection * view * model;
    
    // Use shader program
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
    glm::vec3 objectColor = glm::vec3(0.8f, 0.3f, 0.3f);  // Nice red cloth color
    
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
        
        if (!dataUpdated) {
            std::cerr << "CUDA interop failed, falling back to CPU method" << std::endl;
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
    
    // Draw the cloth
    glDrawElements(GL_TRIANGLES, numIndices, GL_UNSIGNED_INT, 0);
    
    glBindVertexArray(0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);  // Reset to fill mode
    
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
    ImGui::Text("CUDA Interop: %s", cudaInteropEnabled ? "Enabled" : "Disabled");
    ImGui::Text("Data Updated: %s", dataUpdated ? "Yes" : "No");
    
    ImGui::Separator();
    ImGui::Checkbox("Wireframe", &wireframe);
    
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