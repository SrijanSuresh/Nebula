#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Simple hash function for CUDA randomness
__device__ float fast_rand(int seed) {
    int n = seed * 1103515245 + 12345;
    return ((unsigned int)(n / 65536) % 32768) / 32768.0f;
}

__global__ void updateStarPositions(float* pos, float time, int numStars) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numStars) return;
    if (i < numStars) {
        float x = pos[i * 3 + 0];
        float y = pos[i * 3 + 1];
        float z = pos[i * 3 + 2];

        // 1. Apply Swirl
        float theta = 0.005f; 
        pos[i * 3 + 0] = sin(time + i);
        pos[i * 3 + 1] = cos(time + i);

        // 2. Continuous Z Movement
        z += 0.004f; 
        
        // 3. Organic Reset
        if (z > 1.0f) {
            z = -1.0f;
            // Randomize X and Y in range [-1, 1] to break the "box"
            pos[i * 3 + 0] = fast_rand(i + (int)(time * 1000)) * 2.0f - 1.0f;
            pos[i * 3 + 1] = fast_rand(i + 7) * 2.0f - 1.0f;
        }
        pos[i * 3 + 2] = z;
    }
}
	
extern "C" void launch_nebula_kernel(float* d_pos, float time, int numStars) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (numStars + threadsPerBlock - 1) / threadsPerBlock;

    // launch kernel
    updateStarPositions<<<blocksPerGrid, threadsPerBlock>>>(d_pos, time, numStars);

    // synchronize
    cudaDeviceSynchronize();
}
