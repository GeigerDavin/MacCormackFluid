#ifndef OPENGL_INTEROP_HPP
#define OPENGL_INTEROP_HPP

void registerGraphicsResource
    (cudaGraphicsResource** graphicsResource, unsigned int bufferId);
void unregisterGraphicsResource
    (cudaGraphicsResource** graphicsResource);
void* mapGraphicsResource
    (cudaGraphicsResource** graphicsResource);
void unmapGraphicsResource
    (cudaGraphicsResource*  graphicsResource);

#endif