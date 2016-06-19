#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "OpenGLInterop.hpp"
#include "Auxiliary.hpp"

#define WINDOW_HEIGT (640)
#define WINDOW_WIDTH (480)
using namespace std;

int main()
{
    initializeCuda(true);

	GLFWwindow* mainWindow;
	cudaError_t cudaStatus;
	if (!glfwInit())
	{
		cerr << "Failind to Initialize GLFW. Error=" << glGetError() << endl;
		return -1;
	}

	// Create Main Window
	mainWindow = glfwCreateWindow(WINDOW_HEIGT, WINDOW_WIDTH, "Mac Cormack Fluid", NULL, NULL);
	if (!mainWindow)
	{
		cerr << "Failed to Create the Main Window. Error=" << glGetError() << endl;
		glfwTerminate();
		return -1;
	}
	// Register Main Window
	glfwMakeContextCurrent(mainWindow);

	glewExperimental = GL_FALSE;
	if (glewInit() != GLEW_OK)
	{
		cerr << "Failind to Initialize GLEW. Error=" << glGetError() << endl;
		return -1;
	}

	//Init Cuda Device
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaSetDevice failed. Error: " << cudaGetErrorString(cudaGetLastError()) << endl;
		return -1;
	}

	GLuint positionsVBO;
	struct cudaGraphicsResource* positionsVBO_CUDA;

	glGenBuffers(1, &positionsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
	unsigned int size = WINDOW_HEIGT * WINDOW_WIDTH * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsMapFlagsWriteDiscard);

	// Loop until the User closes the Main Window
	while (!glfwWindowShouldClose(mainWindow))
	{
		// Render here 
		glClear(GL_COLOR_BUFFER_BIT);

		// Swap front and back buffers
		glfwSwapBuffers(mainWindow);

		// Poll for and process events
		glfwPollEvents();
	}

	// Exit GLFW Cleanly
	glfwTerminate();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		cerr << "cudaDeviceReset failed. Error: " << cudaGetErrorString(cudaGetLastError()) << endl;
	    return -1;
	}
	return 0;
}