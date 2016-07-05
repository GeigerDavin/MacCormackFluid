#include "StdAfx.hpp"
#include "Camera/Camera.hpp"

#include "CUDA/CudaArray.hpp"
#include "CUDA/CudaArray3D.hpp"
#include "CUDA/CudaTextureObject.hpp"
#include "CUDA/CudaSurfaceObject.hpp"
#include "CUDA/CudaGLGraphicsResource.hpp"
#include "CUDA/Modules/DeviceManagement.hpp"
#include "CUDA/Modules/MemoryManagement.hpp"
#include "CUDA/Modules/TextureReferenceManagement.hpp"

#include "OpenGL/OpenGLBuffer.hpp"
#include "OpenGL/OpenGLShaderProgram.hpp"
#include "OpenGL/OpenGLVertexArrayObject.hpp"

#include "Utils/TrackSphere.hpp"

#include "Kernel/Constant.hpp"
#include "Kernel/Render.hpp"

#define WINDOW_WIDTH (512)
#define WINDOW_HEIGHT (512)
int width = WINDOW_WIDTH;
int height = WINDOW_HEIGHT;
GLFWwindow* mainWindow = nullptr;

#define DIMXYZ 20

const int dimX = DIMXYZ;
const int dimY = DIMXYZ;
const int dimZ = DIMXYZ;

int viewOrientation = 0;
int viewSclice = DIMXYZ / 2;

int mouseButtonState = -1;
int mousePosX = 0;
int mousePosY = 0;

int mousePosXPrev = 0;
int mousePosYPrev = 0;

bool drag = false;
int mousePosXDragStart = 0;
int mousePosYDragStart = 0;
int mousePosXDragEnd = 0;
int mousePosYDragEnd = 0;

float zoom = 1.2f;

Utils::TrackSphere trackSphere;

float3 viewRotation = make_float3(1.0f, 2.0f, 3.0f);
float3 viewTranslation = make_float3(1.0f, 0.0f, -4.0f);

CUDA::CudaTextureObject speedSizeTexture;
CUDA::CudaSurfaceObject speedSizeSurface;

OpenGL::OpenGLBuffer rgbaBuffer;
CUDA::CudaGLGraphicsResource rgbaGraphicsResource;

bool initialize();
void createVolumes();
void createOutputBuffer();
void update();
void render();
void cleanup();

static void cursorPosClb(GLFWwindow* window, double xpos, double ypos);
static void cursorButtonClb(GLFWwindow* window, int button, int action, int mods);
static void cursorScrollClb(GLFWwindow* window, double xoffset, double yoffset);

int main() {
    if (!initialize()) {
        return EXIT_FAILURE;
    }

    createVolumes();
    createOutputBuffer();

    while (!glfwWindowShouldClose(mainWindow)) {
        update();
        render();

        glfwSwapInterval(0);
        glfwSwapBuffers(mainWindow);
        glfwPollEvents();

        if (glfwGetKey(mainWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(mainWindow, true);
        }
    }

    cleanup();
    glfwTerminate();
    cudaDeviceReset();

    return EXIT_SUCCESS;
}
void GLFW_ErrorCallback(int error, const char* description)
{
	  std::cerr << description << std::endl;
}

bool initialize() {
    CUDA::DeviceManagement::initializeCuda(true);
    if (!CUDA::useCuda) {
        std::cerr << "CPU only not supported" << std::endl;
        return false;
    }

    glfwSetErrorCallback(GLFW_ErrorCallback);

    if (!glfwInit()) {
        std::cerr << "Failed to Initialize GLFW. Error=" << glGetError() << std::endl;
        return false;
    }

    mainWindow = glfwCreateWindow(width, height, "MacCormack Fluid", NULL, NULL);
    if (!mainWindow) {
        std::cerr << "Failed to Create the Main Window. Error=" << glGetError() << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(mainWindow);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to Initialize GLEW. Error=" << glGetError() << std::endl;
        return false;
    }

    glfwSetCursorPosCallback(mainWindow, cursorPosClb);
    glfwSetMouseButtonCallback(mainWindow, cursorButtonClb);
    glfwSetScrollCallback(mainWindow, cursorScrollClb);

    return true;
}

void createVolumes() {
    unsigned short  *volume = new  unsigned short[dimX*dimY*dimZ * 4];
    memset(volume, 0, dimX*dimY*dimZ * 2 * 4);

    cudaChannelFormatDesc desc = CUDA::TextureReferenceManagement::createChannelDesc<float>();
    cudaExtent extent = CUDA::MemoryManagement::createCudaExtent(dimX, dimY, dimZ);
    cudaArray_t speedSizeArray = CUDA::MemoryManagement::malloc3DArray(&desc, extent, cudaArraySurfaceLoadStore);

    speedSizeTexture.setResourceType(CUDA::CudaTextureObject::ResourceType::Array);
    speedSizeTexture.setReadMode(CUDA::CudaTextureObject::ReadMode::ElementType);
    speedSizeTexture.setFilterMode(CUDA::CudaTextureObject::FilterMode::PointFilter);
    speedSizeTexture.setAddressMode0(CUDA::CudaTextureObject::AddressMode::Border);
    speedSizeTexture.setAddressMode1(CUDA::CudaTextureObject::AddressMode::Border);
    speedSizeTexture.setAddressMode2(CUDA::CudaTextureObject::AddressMode::Border);
    speedSizeTexture.setNormalized(true);
    if (speedSizeTexture.create(speedSizeArray)) {
        std::cout << "Volume texture object successfully created" << std::endl;
    }

    speedSizeSurface.setResourceType(CUDA::CudaSurfaceObject::ResourceType::Array);
    speedSizeSurface.create(speedSizeArray);


    free(volume);
}

void createOutputBuffer() {
    rgbaBuffer.setType(OpenGL::OpenGLBuffer::Type::PixelUnpackBuffer);
    rgbaBuffer.setUsagePattern(OpenGL::OpenGLBuffer::UsagePattern::StreamDraw);
    rgbaBuffer.create();
    rgbaBuffer.bind();
    rgbaBuffer.allocate(width * height * sizeof(GLubyte) * 4);
    rgbaBuffer.release();

    rgbaGraphicsResource.setRegisterFlag(CUDA::CudaGLGraphicsResource::RegisterFlag::WriteDiscard);
    rgbaGraphicsResource.create(rgbaBuffer.getId());
}

void update() {
    Kernel::Constant constHost;
    constHost.volumeSize.x = dimX;
    constHost.volumeSize.y = dimY;
    constHost.volumeSize.z = dimZ;

    constHost.viewPort.x = width;
    constHost.viewPort.y = height;

    constHost.viewSlice = viewSclice;
    constHost.viewOrientation = viewOrientation;

    constHost.mouse.x = (float) mousePosX;
    constHost.mouse.y = (float) mousePosY;
    constHost.mouse.z = viewSclice;
    constHost.mouse.w = 3;

    if (drag) {
        constHost.dragDirection.x = (float)(mousePosX - mousePosXPrev)*dimX/ width;
        constHost.dragDirection.y = (float)(mousePosY - mousePosYPrev)*dimY/ height;
    } else {
        constHost.dragDirection.x = 0;
        constHost.dragDirection.y = 0;
    }
    constHost.dragDirection.z = 0;

    Utils::Mat3 rot = trackSphere.getRotationMatrix();
    for (int i = 0; i < 3; i++) {
        constHost.rotation.m[i].x = rot(i, 0);
        constHost.rotation.m[i].y = rot(i, 1);
        constHost.rotation.m[i].z = rot(i, 2);
        //std::cout << constHost.rotation.m[i].x;
    }

    constHost.zoom = zoom;

    Kernel::copyToConstant(constHost);

    uint* rgba = (uint *) rgbaGraphicsResource.map();
    CUDA::MemoryManagement::deviceMemset(rgba, 0, width * height * 4);
    Kernel::project3D(0, speedSizeSurface.getSurf(), speedSizeTexture.getTex(), (dimX + 63) / 64, dimY / 4, dimZ / 4);
    Kernel::renderVolume(speedSizeSurface.getSurf(), rgba, (width + 15) / 16, (height + 15)/16, 1);
    rgbaGraphicsResource.unmap();
}

void render() {
    //GLfloat modelView[16];
    //glMatrixMode(GL_MODELVIEW);
    //glPushMatrix();
    //glLoadIdentity();
    //glRotatef(-viewRotation.x, 1.0f, 0.0f, 0.0f);
    //glRotatef(-viewRotation.y, 0.0f, 1.0f, 0.0f);
    //glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    //glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    //glPopMatrix();

    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);

    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    //glEnable(GL_DEPTH_TEST);
    //glEnable(GL_CULL_FACE);
    //glDepthFunc(GL_LESS);

    //glRasterPos2i(0, 0);
    rgbaBuffer.bind();
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    rgbaBuffer.release();
}

void cleanup() {

}

static void cursorPosClb(GLFWwindow* window, double xpos, double ypos) {
    mousePosXPrev = mousePosX;
    mousePosYPrev = mousePosY;

    mousePosX = xpos;
    mousePosY = ypos;

    if (mouseButtonState == GLFW_MOUSE_BUTTON_RIGHT) {
        trackSphere.dragMove(xpos, ypos, width, height);
    }
}

static void cursorButtonClb(GLFWwindow* window, int button, int action, int mods) {
    switch (action) {
    case GLFW_PRESS:
        mouseButtonState = button;
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            double xPos, yPos;
            glfwGetCursorPos(window, &xPos, &yPos);
            trackSphere.dragStart(xPos, yPos, width, height);
        }
        break;
    case GLFW_RELEASE:
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            trackSphere.dragEnd();
        }
        mouseButtonState = -1;
        break;
    }
}

static void cursorScrollClb(GLFWwindow* window, double xoffset, double yoffset) {
    if (yoffset < 0) {
        zoom = zoom * 1.1f;
    } else {
        zoom = zoom / 1.1f;
    }
}
