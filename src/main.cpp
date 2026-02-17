#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <nebula.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

using namespace std;


// importing shader source
// 1. Vector Shader Soruce
const char* vertexShaderSource = R"(
  #version 450 core
  layout (location = 0) in vec3 aPos;
  out float v_z; 
  void main() {
    v_z = aPos.z;
    float zDepth = aPos.z + 1.8; // Further back for better FOV
    gl_Position = vec4(aPos.x / zDepth, aPos.y / zDepth, aPos.z, 1.0);
    gl_PointSize = (aPos.z + 1.0) * 2.5; // Bigger stars when closer
   }    
)";

// 2. Fragment Shader Source
const char* fragmentShaderSource = R"(
  #version 450 core
  out vec4 FragColor;
  in float v_z;
  uniform float u_time;
  void main() {
     float t = (v_z + 1.0) / 2.0;
     vec3 farColor = vec3(0.3, 0.0, 0.5);  // Deep Purple
     vec3 nearColor = vec3(1.0, 0.7, 0.4); // Bright Peach/Gold
     vec3 finalColor = mix(farColor, nearColor, t);
    
     // Smooth circle shape for particles
     float dist = length(gl_PointCoord - vec2(0.5));
     if (dist > 0.5) discard;
    
     FragColor = vec4(finalColor * (1.0 - dist * 2.0), 1.0);
  }

)";

vector<float>generateStars(int count){
	/* we are not using 2d vector since lookup of 2d vector is pretty 
	 inefficient for gpu, so we flatten them into 1d contiguos array.
	 so we have like [x_1,y_1,z_1,....,x_n,y_n,z_n] from this we can
	 iterate position by setting:
	 x index -> i*3, y index -> i*3+1, z index-> i*3+2 */
	
	vector<float>stars;
	stars.reserve(count*3); // x,y,z

	for(int i = 0; i < count; i++){
	    float x = (float)rand()/(float)RAND_MAX * 2.0f - 1.0f;
	    float y = (float)rand()/(float)RAND_MAX * 2.0f - 1.0f;
	    float z = (float)rand()/(float)RAND_MAX * 2.0f - 1.0f;
	    stars.push_back(x);
	    stars.push_back(y);
	    stars.push_back(z);
	}
	return stars;
}
// cuda/opengl interoperability
struct cudaGraphicsResource* cuda_vbo_resource;

int main(){

    // graphics lib framework init
    if (!glfwInit()){
        return -1;
    }

    // window setup
    GLFWwindow* window = glfwCreateWindow(1600, 900, "WindowGL", NULL, NULL);
    
    if (window == NULL){
      	glfwTerminate();
      	return -1;
    }
    glfwMakeContextCurrent(window);
    
    // loading glad to retrieve latest runtimes for glfw
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
      	return -1;
    }

    // vertices for stars
    vector<float> vertices = generateStars(100000);

    // create a vector array object and buffer obj then bind it to GPU
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    // creating buffer data
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(float), vertices.data(), GL_STATIC_DRAW);
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsRegisterFlagsWriteDiscard);
    
    if (err != cudaSuccess) {
      printf("cudaGraphicsGLRegisterBuffer failed: %s\n", cudaGetErrorString(err));
      // You can glfwTerminate() + return -1; here
      glfwTerminate();
      return -1;
    }
    // read binary data for vbo
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    // unbind after setup is done
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // SHADER LOGIC
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER); // vertex shader
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER); // fragment shader

    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    
    glCompileShader(vertexShader); glCompileShader(fragmentShader); // compiler

    unsigned int shaderProgram = glCreateProgram(); // linker
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // get u_time back from gpu
    int timeLoc = glGetUniformLocation(shaderProgram, "u_time");
    
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // to make star pulsing to look like fade
    glEnable(GL_PROGRAM_POINT_SIZE); // WSL shader change fix
    // window loop
    while(!glfwWindowShouldClose(window)){
    	// clear screen
    	glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    	glClear(GL_COLOR_BUFFER_BIT);
        
	// CUDA
	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);

	float* d_vbo_ptr; size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, cuda_vbo_resource);
	
	// kernel launch will go here
	float timeVal = glfwGetTime(); 
	if (num_bytes >= 100000 * 3 * sizeof(float)) {
    	    launch_nebula_kernel(d_vbo_ptr, (float)glfwGetTime(), 100000);
	}
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
    	   printf("CUDA Error: %s\n", cudaGetErrorString(err));
	}
	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
	
	// attach shaders then draw
	glBindVertexArray(vao);
	glUseProgram(shaderProgram);
	float time_val = glfwGetTime(); // geting uniform time for star pulsing
	glUniform1f(timeLoc, time_val);
	glDrawArrays(GL_POINTS, 0, 100000);

    	
	// swap and poll
    	glfwSwapBuffers(window);
    	glfwPollEvents();
    }
    // unbind vector array object
    glBindVertexArray(0);

    // delete shaders after use
    glDeleteShader(vertexShader); glDeleteShader(fragmentShader);

    glfwTerminate();
    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    return 0;

}
