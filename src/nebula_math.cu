#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

// Simple hash function for CUDA randomness
__device__ float fast_rand(int seed) {
    int n = seed * 1103515245 + 12345;
    return ((unsigned int)(n / 65536) % 32768) / 32768.0f;
}

__global__ void updateStarPositions(float* pos, float time, int numStars) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numStars) return;

    float x = pos[i * 3 + 0];
    float y = pos[i * 3 + 1];
    float z = pos[i * 3 + 2];

    // Distance-based rotation speed (Differential Rotation)
    float dist = sqrt(x*x + y*y);
    float theta = 0.01f / (dist + 0.1f); 
    
    pos[i * 3 + 0] = x * cos(theta) - y * sin(theta);
    pos[i * 3 + 1] = x * sin(theta) + y * cos(theta);

    // Subtle drifting in Z
    z += 0.002f * sinf(time * 0.5f + i * 0.001f);
    if (z > 0.4f) z = -0.4f;
    pos[i * 3 + 2] = z;
}

extern "C" void launch_nebula_kernel(float* d_pos, float time, int numStars) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numStars + threadsPerBlock - 1) / threadsPerBlock;

    // launch kernel
    updateStarPositions<<<blocksPerGrid, threadsPerBlock>>>(d_pos, time, numStars);

    // synchronize
    cudaDeviceSynchronize();
}
