#include "StdAfx.hpp"
#include "Utils/Vector.hpp"
#include "CUDA/Auxiliary.hpp"

#include "OpenGL/OpenGLBuffer.hpp"
#include "OpenGL/OpenGLShaderProgram.hpp"
#include "OpenGL/OpenGLVertexArrayObject.hpp"

#include "Kernel/MacCormack.hpp"

#define WINDOW_WIDTH (1280)
#define WINDOW_HEIGHT (720)

static const std::vector<GLfloat> g_vertex_buffer_data = {
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

static const std::vector<GLfloat> g_color_buffer_data = {
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
    CUDA::initializeCuda(true);
    if (!CUDA::useCuda) {
        std::cerr << "CPU only not supported" << std::endl;
        return -1;
    }

	GLFWwindow* mainWindow = nullptr;
	if (!glfwInit())
	{
		std::cerr << "Failind to Initialize GLFW. Error=" << glGetError() << std::endl;
		return -1;
	}

    int width = WINDOW_WIDTH;
    int height = WINDOW_HEIGHT;
	mainWindow = glfwCreateWindow(width, height, "MacCormack Fluid", NULL, NULL);
	if (!mainWindow)
	{
		std::cerr << "Failed to Create the Main Window. Error=" << glGetError() << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(mainWindow);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		std::cerr << "Failind to Initialize GLEW. Error=" << glGetError() << std::endl;
		return -1;
	}

    OpenGL::OpenGLShaderProgram shaderProgram;
    OpenGL::OpenGLVertexArrayObject vao;

	vao.create();
	vao.bind();

    shaderProgram.addShaderFromSourceFile(OpenGL::OpenGLShader::Vertex, "Shader/CubeVertex.glsl");
    shaderProgram.addShaderFromSourceFile(OpenGL::OpenGLShader::Fragment, "Shader/CubeFragment.glsl");
    shaderProgram.link();

    Utils::Vector<GLfloat> vertices(&g_vertex_buffer_data[0], g_vertex_buffer_data.size());
    Utils::Vector<GLfloat> colors(&g_color_buffer_data[0], g_color_buffer_data.size());

    OpenGL::OpenGLBuffer* buffer = vertices.getBuffer();
    buffer->bind();
    shaderProgram.enableAttributeArray(0);
    shaderProgram.setAttributeBuffer(0, GL_FLOAT, 0, 3, 0);
    buffer->release();

    buffer = colors.getBuffer();
    buffer->bind();
    shaderProgram.enableAttributeArray(1);
    shaderProgram.setAttributeBuffer(1, GL_FLOAT, 0, 3, 0);
    buffer->release();

	vao.release();

    GLuint MatrixID = shaderProgram.getUniformLocation("MVP");
	while (!glfwWindowShouldClose(mainWindow))
	{
        Kernel::computeCube(vertices, colors);

        glfwGetWindowSize(mainWindow, &width, &height);
        glfwGetFramebufferSize(mainWindow, &width, &height);
        glViewport(0, 0, width, height);

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)width / height, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(glm::vec3(4, 3, -3),
                                     glm::vec3(0, 0, 0),
                                     glm::vec3(0, 1, 0));
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 MVP = projection * view * model;

        shaderProgram.bind();
        shaderProgram.setUniformValue(MatrixID, MVP);
		{
			OpenGL::OpenGLVertexArrayObject::Binder binder(&vao);
			glDrawArrays(GL_TRIANGLES, 0, 12 * 3);

		}
		shaderProgram.release();

		glfwSwapBuffers(mainWindow);
		glfwPollEvents();
	}

	glfwTerminate();

	cudaDeviceReset();
    ERRORCHECK_CUDA();

	return 0;
}