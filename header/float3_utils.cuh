#ifndef FLOAT3_UTILS_CUH
#define FLOAT3_UTILS_CUH

#include <cuda_runtime.h>
#include <math.h>
#include <cstdio>

__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ inline float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ inline float3 operator*(const float3& a, const float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator*(const float s, const float3& a) {
    return a * s;
}

__host__ __device__ inline float3 operator/(const float3& a, const float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

__host__ __device__ inline float3& operator*=(float3& a, const float3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    return a;
}

__host__ __device__ inline float3& operator*=(float3& a, const float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
}

__host__ __device__ inline float3& operator/=(float3& a, const float s) {
    a.x /= s;
    a.y /= s;
    a.z /= s;
    return a;
}

__host__ __device__ inline float3& operator/=(float3& a, const float3& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    return a;
}

__host__ __device__ inline float length(const float3& a) {
    return sqrtf(dot(a, a));
}

__host__ __device__ inline float3 normalize(const float3& a) {
    float len = length(a);
    return (len > 1e-8f) ? a / len : make_float3(0.0f, 0.0f, 0.0f);
}

__host__ __device__ inline float3 clamp(const float3& a, float minVal, float maxVal) {
    return make_float3(
        fminf(fmaxf(a.x, minVal), maxVal),
        fminf(fmaxf(a.y, minVal), maxVal),
        fminf(fmaxf(a.z, minVal), maxVal)
    );
}

__host__ __device__ inline void print_float3(const char* label, const float3& v) {
#if !defined(__CUDA_ARCH__) // only on host
    printf("%s: (%.4f, %.4f, %.4f)\n", label, v.x, v.y, v.z);
#endif
}

#endif // FLOAT3_UTILS_CUH
