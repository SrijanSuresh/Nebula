#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
using namespace std;

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

    // vertices for triangle
    float vertices[] = {
    	-0.5f, -0.5f, 0.0f, // bottom-left
    	0.5f, -0.5f, 0.0f, // bottom-right
	0.0f, 0.5f, 0.0f // up
    
    }; 
    // create a vector array object and buffer obj then bind it to GPU
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    // creating buffer data
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    // read binary data for vbo
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    // unbind after setup is done
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // window loop
    while(!glfwWindowShouldClose(window)){
    	// clear screen
    	glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    	glClear(GL_COLOR_BUFFER_BIT);
        
	// draw
	glBindVertexArray(vao);
	glDrawArrays(GL_TRIANGLES, 0, 3);

    	
	// swap and poll
    	glfwSwapBuffers(window);
    	glfwPollEvents();
    }
    // unbind vector array object
    glBindVertexArray(0);

    glfwTerminate();
    return 0;

}
