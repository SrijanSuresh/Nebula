#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <nebula.h>
#include <iostream>
#include <ctime>
#include <cmath>
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
    gl_PointSize = 4.0 + (2.0 * (1.0 - v_z)); // Larger points for "gas" feel   
  }    
)";

// 2. Fragment Shader Source
const char* fragmentShaderSource = R"(
  #version 450 core
  out vec4 FragColor;
  in float v_z;
  uniform float u_time;
  void main() {
  // Calculate distance from the center of the point (0.0 to 0.5)
    float dist = distance(gl_PointCoord, vec2(0.5));
    
    // Create a "Soft" falloff (Gaussian-like)
    // The further from center, the lower the alpha.
    float alpha = exp(-dist * dist * 15.0); 

    // Define Nebula Colors
    vec3 coreColor = vec3(1.0, 0.8, 0.6); // Warm Peach
    vec3 edgeColor = vec3(0.5, 0.2, 0.8); // Deep Purple
    
    // Mix color based on depth (Z) and distance from center
    vec3 color = mix(edgeColor, coreColor, (v_z + 0.5));
    
    // Multiply alpha by a tiny factor so 1M stars don't just turn white
    FragColor = vec4(color, alpha * 0.15);
  }

)";


vector<float> generateStars(int count) {
    vector<float> stars;
    stars.reserve(count * 3);
    for (int i = 0; i < count; i++) {
        // Box-Muller transform for Gaussian distribution
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        float radius = sqrt(-2.0f * log(u1)) * 0.4f; // Controls "spread"
        float angle = 2.0f * 3.14159f * u2;

        float x = radius * cos(angle);
        float y = radius * sin(angle);
        float z = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.2f; // Keep it somewhat flat

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
    GLFWwindow* window = glfwCreateWindow(1920, 1080, "WindowGL", NULL, NULL);
    
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
    vector<float> vertices = generateStars(1000000);

    // create a vector array object and buffer obj then bind it to GPU
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    // creating buffer data
    float* d_nebula_ptr; // This is our dedicated CUDA buffer
    size_t bufferSize = 1000000 * 3 * sizeof(float);

    // Allocate memory directly on the GPU
    cudaMalloc(&d_nebula_ptr, bufferSize);

    // Initialize CUDA buffer with the starting positions
    cudaMemcpy(d_nebula_ptr, vertices.data(), bufferSize, cudaMemcpyHostToDevice);
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
    
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
	float timeVal = (float)glfwGetTime();

    	// 1. Run the physics kernel on our dedicated CUDA memory
    	launch_nebula_kernel(d_nebula_ptr, timeVal, 1000000);

    	// 2. Manual Bridge: Move data from CUDA buffer to OpenGL buffer
   	 glBindBuffer(GL_ARRAY_BUFFER, vbo);
   	 void* gl_ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    
    	// Copy updated positions into the memory OpenGL uses to draw
    	cudaMemcpy(gl_ptr, d_nebula_ptr, bufferSize, cudaMemcpyDeviceToHost);
    
   	 glUnmapBuffer(GL_ARRAY_BUFFER);	
	// attach shaders then draw
	glBindVertexArray(vao);
	glUseProgram(shaderProgram);
	float time_val = glfwGetTime(); // geting uniform time for star pulsing
	glUniform1f(timeLoc, time_val);
	glDrawArrays(GL_POINTS, 0, 1000000);

    	
	// swap and poll
    	glfwSwapBuffers(window);
    	glfwPollEvents();
    }
    // unbind vector array object
    glBindVertexArray(0);

    // delete shaders after use
    glDeleteShader(vertexShader); glDeleteShader(fragmentShader);

    glfwTerminate();
    cudaFree(d_nebula_ptr);
    return 0;

}
