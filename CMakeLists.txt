cmake_minimum_required(VERSION 3.18)
project(RockIntoClothSim LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Project directories
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(HEADER_DIR ${PROJECT_SOURCE_DIR}/header)
set(IMGUI_DIR $ENV{IMGUI_DIR} CACHE PATH "Path to ImGui")
if(NOT IMGUI_DIR)
    set(IMGUI_DIR $ENV{HOME}/imgui)
endif()

# Find packages
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)
find_package(GLEW REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(glm REQUIRED)  # Add GLM package find

# ImGui sources
set(IMGUI_SRC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

# Project sources
set(SRC_FILES
    ${SRC_DIR}/main.cu
    ${SRC_DIR}/cloth.cu
    ${SRC_DIR}/params.cu
    ${SRC_DIR}/integrator.cu
    ${SRC_DIR}/applied_force.cu
    ${SRC_DIR}/energy.cu
    ${SRC_DIR}/convergence.cu
    ${SRC_DIR}/extract_positions.cu
    ${SRC_DIR}/renderer.cpp
    ${SRC_DIR}/vtk_export.cpp
    ${IMGUI_SRC}
)

add_executable(cloth_sim ${SRC_FILES})

# Include directories (PRIVATE to target)
target_include_directories(cloth_sim PRIVATE
    ${HEADER_DIR}
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${CUDAToolkit_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${glm_INCLUDE_DIRS}  # Add GLM include directories
)

# Link libraries (PRIVATE to target)
target_link_libraries(cloth_sim PRIVATE
    OpenGL::GL
    glfw
    GLEW::GLEW
    dl
    pthread
    glm::glm  
    CUDA::cudart
)

# CUDA properties
set_target_properties(cloth_sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Optional: Build Toggle for ImGui
option(ENABLE_IMGUI "Build with ImGui GUI" ON)
if(NOT ENABLE_IMGUI)
    target_compile_definitions(cloth_sim PRIVATE NO_IMGUI)
endif()

# Build configuration messages
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
else()
    message(STATUS "Building in Release mode")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
endif()