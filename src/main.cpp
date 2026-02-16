#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
using namespace std;


// importing shader source
// 1. Vector Shader Soruce
const char* vertexShaderSource = R"(
    #version 450 core
    layout (location = 0) in vec3 aPos;
    uniform float u_time; // retrieve uniform time from cpu to shader
    void main() {
        gl_Position = vec4(aPos, 1.0);
	gl_PointSize = 2.0;
    }
)";

// 2. Fragment Shader Source
const char* fragmentShaderSource = R"(
    #version 450 core
    out vec4 FragColor;
    uniform float u_time;
    void main() {
	float offset = gl_FragCoord.x * 0.01 + gl_FragCoord.y * 0.01; // unique offset for each star
	float pulse = (sin(u_time * 2.5 + offset) + 1.0) / 2.0; // oscillation between 0 and 1
        FragColor = vec4(1.0, 0.8, 0.5, pulse); // yellow-orange color
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


int main(){

    // graphics lib framework init
    if (!glfwInit()){
        return -1;
    }

    // window setup
    GLFWwindow* window = glfwCreateWindow(800, 600, "WindowGL", NULL, NULL);
    
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
    vector<float> vertices = generateStars(1000);

    // create a vector array object and buffer obj then bind it to GPU
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    // creating buffer data
    glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(float), vertices.data(), GL_STATIC_DRAW);
    
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
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // to make star pulsing to look like fade
    glEnable(GL_PROGRAM_POINT_SIZE); // WSL shader change fix
    // window loop
    while(!glfwWindowShouldClose(window)){
    	// clear screen
    	glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    	glClear(GL_COLOR_BUFFER_BIT);
        
	// attach shaders then draw
	glBindVertexArray(vao);
	glUseProgram(shaderProgram);
	float timeVal = glfwGetTime(); // geting uniform time for star pulsing
	glUniform1f(timeLoc, timeVal);
	glDrawArrays(GL_POINTS, 0, 1000);

    	
	// swap and poll
    	glfwSwapBuffers(window);
    	glfwPollEvents();
    }
    // unbind vector array object
    glBindVertexArray(0);

    // delete shaders after use
    glDeleteShader(vertexShader); glDeleteShader(fragmentShader);

    glfwTerminate();
    return 0;

}
