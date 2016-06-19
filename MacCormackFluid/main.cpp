#include "StdAfx.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "OpenGLInterop.hpp"
#include "Auxiliary.hpp"
#include "OpenGLBuffer.hpp"
#include "OpenGLShaderProgram.hpp"
#include "OpenGLVertexArrayObject.hpp"


#define WINDOW_HEIGT (640)
#define WINDOW_WIDTH (480)
using namespace std;
static const GLfloat g_vertex_buffer_data[] = {
	-1.0f, -1.0f, -1.0f,
	-1.0f, -1.0f, 1.0f,
	-1.0f, 1.0f, 1.0f,
	1.0f, 1.0f, -1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f, 1.0f, -1.0f,
	1.0f, -1.0f, 1.0f,
	-1.0f, -1.0f, -1.0f,
	1.0f, -1.0f, -1.0f,
	1.0f, 1.0f, -1.0f,
	1.0f, -1.0f, -1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f, 1.0f, 1.0f,
	-1.0f, 1.0f, -1.0f,
	1.0f, -1.0f, 1.0f,
	-1.0f, -1.0f, 1.0f,
	-1.0f, -1.0f, -1.0f,
	-1.0f, 1.0f, 1.0f,
	-1.0f, -1.0f, 1.0f,
	1.0f, -1.0f, 1.0f,
	1.0f, 1.0f, 1.0f,
	1.0f, -1.0f, -1.0f,
	1.0f, 1.0f, -1.0f,
	1.0f, -1.0f, -1.0f,
	1.0f, 1.0f, 1.0f,
	1.0f, -1.0f, 1.0f,
	1.0f, 1.0f, 1.0f,
	1.0f, 1.0f, -1.0f,
	-1.0f, 1.0f, -1.0f,
	1.0f, 1.0f, 1.0f,
	-1.0f, 1.0f, -1.0f,
	-1.0f, 1.0f, 1.0f,
	1.0f, 1.0f, 1.0f,
	-1.0f, 1.0f, 1.0f,
	1.0f, -1.0f, 1.0f
};

// One color for each vertex. They were generated randomly.
static const GLfloat g_color_buffer_data[] = {
	0.583f, 0.771f, 0.014f,
	0.609f, 0.115f, 0.436f,
	0.327f, 0.483f, 0.844f,
	0.822f, 0.569f, 0.201f,
	0.435f, 0.602f, 0.223f,
	0.310f, 0.747f, 0.185f,
	0.597f, 0.770f, 0.761f,
	0.559f, 0.436f, 0.730f,
	0.359f, 0.583f, 0.152f,
	0.483f, 0.596f, 0.789f,
	0.559f, 0.861f, 0.639f,
	0.195f, 0.548f, 0.859f,
	0.014f, 0.184f, 0.576f,
	0.771f, 0.328f, 0.970f,
	0.406f, 0.615f, 0.116f,
	0.676f, 0.977f, 0.133f,
	0.971f, 0.572f, 0.833f,
	0.140f, 0.616f, 0.489f,
	0.997f, 0.513f, 0.064f,
	0.945f, 0.719f, 0.592f,
	0.543f, 0.021f, 0.978f,
	0.279f, 0.317f, 0.505f,
	0.167f, 0.620f, 0.077f,
	0.347f, 0.857f, 0.137f,
	0.055f, 0.953f, 0.042f,
	0.714f, 0.505f, 0.345f,
	0.783f, 0.290f, 0.734f,
	0.722f, 0.645f, 0.174f,
	0.302f, 0.455f, 0.848f,
	0.225f, 0.587f, 0.040f,
	0.517f, 0.713f, 0.338f,
	0.053f, 0.959f, 0.120f,
	0.393f, 0.621f, 0.362f,
	0.673f, 0.211f, 0.457f,
	0.820f, 0.883f, 0.371f,
	0.982f, 0.099f, 0.879f
};
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

	glewExperimental = GL_TRUE;
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

	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);



	OpenGLBuffer bufferColor;
	OpenGLBuffer bufferVertex;
	OpenGLShaderProgram shader;
	OpenGLVertexArrayObject vertexes;


	vertexes.create();
	vertexes.bind();

	shader.addShaderFromSourceFile(OpenGLShader::Vertex, "shader/cubeshader.vertexshader");
	shader.addShaderFromSourceFile(OpenGLShader::Fragment, "shader/cubeshader.fragmentshader");
	shader.link();
	shader.bind();


	bufferColor.create();
	bufferColor.bind();
	bufferColor.allocate(g_vertex_buffer_data, sizeof(g_color_buffer_data));

	// 1rst attribute buffer : vertices
	shader.enableAttributeArray(0);
	shader.setAttributeBuffer(
		0,
		GL_FLOAT,
		0,
		3,
		0
		);

	bufferColor.release();

	bufferVertex.create();
	bufferVertex.bind();
	bufferVertex.allocate(g_vertex_buffer_data, sizeof(g_vertex_buffer_data));


	// 2nd attribute buffer : colors
	shader.enableAttributeArray(1);
	shader.setAttributeBuffer(
		1,
		GL_FLOAT,
		0,
		3,
		0
		);
	bufferVertex.release();

	// Get a handle for our "MVP" uniform
	GLuint MatrixID = shader.getUniformLocation("MVP");

	// Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);
	// Camera matrix
	glm::mat4 View = glm::lookAt(
		glm::vec3(4, 3, -3), // Camera is at (4,3,-3), in World Space
		glm::vec3(0, 0, 0), // and looks at the origin
		glm::vec3(0, 1, 0)  // Head is up (set to 0,-1,0 to look upside-down)
		);
	// Model matrix : an identity matrix (model will be at the origin)
	glm::mat4 Model = glm::mat4(1.0f);
	// Our ModelViewProjection : multiplication of our 3 matrices
	glm::mat4 MVP = Projection * View * Model; // Remember, matrix multiplication is the other way around

	// in the "MVP" uniform
	shader.setUniformValue(MatrixID,MVP);

	vertexes.release();

	////GLuint positionsVBO;
	//struct cudaGraphicsResource* positionsVBO_CUDA;

	//glGenBuffers(1, &positionsVBO);
	//glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
	//unsigned int size = WINDOW_HEIGT * WINDOW_WIDTH * 4 * sizeof(float);
	//glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	//glBindBuffer(GL_ARRAY_BUFFER, 0);
	//cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsMapFlagsWriteDiscard);

	// Loop until the User closes the Main Window
	while (!glfwWindowShouldClose(mainWindow))
	{
		// Render here 
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		shader.bind();
		{
			OpenGLVertexArrayObject::Binder binder(&vertexes);
			// Draw the triangle !
			glDrawArrays(GL_TRIANGLES, 0, 12 * 3); // 12*3 indices starting at 0 -> 12 triangles

		}
		shader.release();


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

void render()
{

	
}