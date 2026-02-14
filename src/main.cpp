#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
using namespace std;

int main(){

    if (!glfwInit()){
        return -1;
    }
    
    GLFWwindow* window = glfwCreateWindow(800, 600, "WindowGL", NULL, NULL);
    
    if (window == NULL){
      glfwTerminate();
      return -1;
    }
    glfwMakeContextCurrent(window);
    
    if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)){
      return -1;
    }
    
    while(!glfwWindowShouldClose(window)){
    // clear screen
    glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // swap and poll
    glfwSwapBuffers(window);
    glfwPollEvents();
    
    }

    glfwTerminate();
    return 0;

}
