#ifndef OPENGL_INTEROP_HPP
#define OPENGL_INTEROP_HPP

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void registerGraphicsResource
    (cudaGraphicsResource** graphicsResource, unsigned int bufferId);
void unregisterGraphicsResource
    (cudaGraphicsResource** graphicsResource);
void* mapGraphicsResource
    (cudaGraphicsResource** graphicsResource);
void unmapGraphicsResource
    (cudaGraphicsResource*  graphicsResource);

#endif