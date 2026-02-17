#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>



__global__ void updateStarPositions(float* pos, float time, int numStars){
/*
	Equation of motion => P_new = P_old + (V*dt)
	Swirl Equation => Centripetal force
	=> F = Force(P_i, time)
	=> V_new = V_old + (F*dt)
 */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	
    if (i < numStars){
	// lets obtain initial positions of x and y
	float x = pos[i * 3];
	float y = pos[i * 3 + 1];		
    	// centripetal swirl, rotate each position by a small angle
	float theta = 0.01f;
	pos[i*3] = x*cos(theta) - y*sin(theta); // adjust angle x
	pos[i*3+1] = x*sin(theta) + y*cos(theta); // adjust angle y
    	// warp speed
	pos[i*3+2] += 0.005f; // adjust z speed
	if (pos[i*3+2] > 1.0f){
	    pos[i*3+2] = -1.0f; // warp reset back to start
	}
    
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
