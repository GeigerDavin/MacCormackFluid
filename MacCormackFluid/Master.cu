#include "StdAfx.hpp"
#include "Master.hpp"

#include "OpenGL/OpenGLBuffer.hpp"
#include "Camera/SimpleRotationCamera.hpp"
#include "CUDA/CudaGLGraphicsResource.hpp"

#include "Kernel/Render.hpp"

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720

class MasterPrivate {
public:
    MasterPrivate(int rank, int worldSize)
        : mpiRank(rank)
        , mpiWorldSize(worldSize)
        , inited(false)
        , windowWidth(WINDOW_WIDTH)
        , windowHeight(WINDOW_HEIGHT)
        , window(nullptr)
        , mouseButtonState(-1)
        , drag(false)
        , zoom(1.2f) {

        instance = this;

        memset(&mousePos, 0, sizeof(int2));
        memset(&mousePosPrev, 0, sizeof(int2));
        memset(&mousePosDragEnd, 0, sizeof(int2));
        memset(&mousePosDragStart, 0, sizeof(int2));

        memset(&volumeSize, 0, sizeof(uint3));
    }

    ~MasterPrivate() {
        destroy();
    }

    bool initialize();
    void destroy();
    void update();
    void render();

    void keyboardEvent(GLFWwindow*, int, int, int, int);
    void mouseButtonEvent(GLFWwindow*, int, int, int);
    void mouseMoveEvent(GLFWwindow*, double, double);
    void mouseWheelEvent(GLFWwindow*, double, double);
    void resizeEvent(GLFWwindow*, int, int);

    static void keyboardEventClb(GLFWwindow*, int, int, int, int);
    static void mouseButtonEventClb(GLFWwindow*, int, int, int);
    static void mouseMoveEventClb(GLFWwindow*, double, double);
    static void mouseWheelEventClb(GLFWwindow*, double, double);
    static void resizeEventClb(GLFWwindow*, int, int);

public:
    int mpiRank;
    int mpiWorldSize;

    bool inited;
    int windowWidth;
    int windowHeight;
    GLFWwindow* window;

    int2 mousePos;
    int2 mousePosPrev;
    int mouseButtonState;

    bool drag;
    int2 mousePosDragStart;
    int2 mousePosDragEnd;

    float zoom;

    Camera::SimpleRotationCamera camera;

    uint3 volumeSize;
    CUDA::CudaArray3D<float> speedSizeArray;
    CUDA::CudaSurfaceObject speedSizeSurface;
    CUDA::CudaTextureObject speedSizeTexture;

    OpenGL::OpenGLBuffer rgbaBuffer;
    CUDA::CudaGLGraphicsResource rgbaGraphicsResource;

private:
    static MasterPrivate* instance;
};

namespace {
    void glfwErrorCallback(int error, const char* description) {
        std::cerr << "[MASTER]: GLFW Error " << error << ": " << description << std::endl;
    }
}

bool MasterPrivate::initialize() {
    if (inited) {
        return true;
    }

    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
        std::cerr << "[MASTER]: Failed to initialize GLFW. Error = " << glGetError() << std::endl;
        return false;
    }

    window = glfwCreateWindow(windowWidth, windowHeight, "MacCormack", NULL, NULL);
    if (!window) {
        std::cerr << "[MASTER]: Failed to create glWindow. Error = " << glGetError() << std::endl;
        return false;
    }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "[MASTER]: Failed to initialize GLEW. Error = " << glGetError() << std::endl;
        return false;
    } else {
        printf("_________________________________________\n");
        std::cout << "[MASTER]: OpenGL version: " << glGetString(GL_VERSION) << std::endl;
        std::cout << "[MASTER]: GLEW version: " << glewGetString(GLEW_VERSION) << std::endl;
        printf("_________________________________________\n");
        printf("\n");
    }

    glfwSetCursorPosCallback(window, mouseMoveEventClb);
    glfwSetMouseButtonCallback(window, mouseButtonEventClb);
    glfwSetScrollCallback(window, mouseWheelEventClb);
    glfwSetWindowSizeCallback(window, resizeEventClb);

    speedSizeArray.create(volumeSize.x, volumeSize.y, volumeSize.z);
    speedSizeTexture.setFilterMode(CUDA::CudaTextureObject::FilterMode::LinearFilter);
    speedSizeTexture.setNormalized(true);
    if (speedSizeTexture.create(speedSizeArray.get())) {
        std::cout << "[MASTER]: Speed size texture object " << speedSizeTexture.getId() << " successfully created" << std::endl;
    } else {
        return false;
    }
    if (speedSizeSurface.create(speedSizeArray.get())) {
        std::cout << "[MASTER]: Speed size surface object " << speedSizeSurface.getId() << " successfully created" << std::endl;
    } else {
        return false;
    }

    rgbaBuffer.setType(OpenGL::OpenGLBuffer::Type::PixelUnpackBuffer);
    rgbaBuffer.setUsagePattern(OpenGL::OpenGLBuffer::UsagePattern::StreamDraw);
    rgbaBuffer.create();
    rgbaBuffer.bind();
    rgbaBuffer.allocate(windowWidth * windowHeight * sizeof(GLubyte) * 4);
    rgbaBuffer.release();

    rgbaGraphicsResource.setRegisterFlag(CUDA::CudaGLGraphicsResource::RegisterFlag::WriteDiscard);
    rgbaGraphicsResource.create(rgbaBuffer.getId());

    float4 volumeSizeHost = make_float4(volumeSize.x, volumeSize.y, volumeSize.z, 0.0f);
    CUDA::MemoryManagement::moveHostToSymbol(volumeSizeDev, volumeSizeHost);

    std::cout << std::endl;

    inited = true;
    return inited;
}

void MasterPrivate::destroy() {
    std::cout << "[MASTER]: Destroying textures..." << std::endl;
    speedSizeTexture.destroy();
    std::cout << "[MASTER]: Textures successfully destroyed" << std::endl << std::endl;

    std::cout << "[MASTER]: Destroying surfaces..." << std::endl;
    speedSizeSurface.destroy();
    std::cout << "[MASTER]: Surfaces successfully destroyed" << std::endl << std::endl;

    std::cout << "[MASTER]: Destroying CUDA arrays..." << std::endl;
    speedSizeArray.destroy();
    std::cout << "[MASTER]: CUDA arrays successfully destroyed" << std::endl << std::endl;

    std::cout << "[MASTER]: Destroying view output..." << std::endl;
    rgbaBuffer.destroy();
    rgbaGraphicsResource.destroy();
    std::cout << "[MASTER]: View output successfully destroyed" << std::endl << std::endl;

    if (window) {
        std::cout << "[MASTER]: Destroying main window..." << std::endl;
        glfwDestroyWindow(window);
        window = nullptr;
        std::cout << "[MASTER]: Main window successfully destroyed" << std::endl << std::endl;
    }
    glfwTerminate();
}

void MasterPrivate::update() {
	mpiBoardCastData.sharedDataGPUHost.viewPort.x = windowWidth;
	mpiBoardCastData.sharedDataGPUHost.viewPort.y = windowHeight;

	mpiBoardCastData.sharedDataGPUHost.viewSlice = 50;
	mpiBoardCastData.sharedDataGPUHost.viewOrientation = 0;

	mpiBoardCastData.sharedDataGPUHost.mouse.x = (float)mousePos.x;
	mpiBoardCastData.sharedDataGPUHost.mouse.y = (float)mousePos.y;
	mpiBoardCastData.sharedDataGPUHost.mouse.z = 50;
	mpiBoardCastData.sharedDataGPUHost.mouse.w = 5.0f;

    if (drag) {
    	mpiBoardCastData.sharedDataGPUHost.dragDirection.x = (float)(mousePos.x - mousePosPrev.x) * volumeSize.x / windowWidth;
    	mpiBoardCastData.sharedDataGPUHost.dragDirection.y = (float)(mousePos.y - mousePosPrev.y) * volumeSize.y / windowHeight;
    } else {
    	mpiBoardCastData.sharedDataGPUHost.dragDirection.x = 0;
    	mpiBoardCastData.sharedDataGPUHost.dragDirection.y = 0;
    }
    mpiBoardCastData.sharedDataGPUHost.dragDirection.z = 0;

    Math::Matrix3x3 rot = camera.getRotationMatrix();
    for (int i = 0; i < 3; i++) {
    	mpiBoardCastData.sharedDataGPUHost.rotation.m[i].x = rot(i, 0);
    	mpiBoardCastData.sharedDataGPUHost.rotation.m[i].y = rot(i, 1);
    	mpiBoardCastData.sharedDataGPUHost.rotation.m[i].z = rot(i, 2);
    }
    mpiBoardCastData.sharedDataGPUHost.zoom = zoom;
    CUDA::MemoryManagement::moveHostToSymbol(g, mpiBoardCastData.sharedDataGPUHost);

    uint* rgba = (uint *)rgbaGraphicsResource.map();
    CUDA::MemoryManagement::deviceMemset(rgba, 0, windowWidth * windowHeight * 4);

    dim3 blockSize = dim3(16, 16, 1);
    dim3 gridSize = dim3((windowWidth + 15) / 16, (windowHeight + 15) / 16, 1);

    Kernel::renderVolume<<<gridSize, blockSize>>>(speedSizeTexture.getId(), rgba);
    getLastCudaError("renderVolume kernel failed");
    //Ctx->synchronize();

    rgbaGraphicsResource.unmap();
}

void MasterPrivate::render() {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
        mpiBoardCastData.sharedDataCPUHost.running = false;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    rgbaBuffer.bind();
    glDrawPixels(windowWidth, windowHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    rgbaBuffer.release();

    mpiBoardCastData.sharedDataCPUHost.elapsedTimeSinceSecond += mpiBoardCastData.sharedDataCPUHost.timeDiff;
    if (mpiBoardCastData.sharedDataCPUHost.elapsedTimeSinceSecond > 1000) {
        char title[256];
        sprintf(title, "MacCormack      Delta Time: %.2fms       Average Time: %.2fms        Total Memory: %lluMB        Free Memory: %lluMB       %.0fFPS",
        		mpiBoardCastData.sharedDataCPUHost.timeDiff,
        		mpiBoardCastData.sharedDataCPUHost.timeAverage,
            Ctx->getTotalDeviceMemoryMegaBytes(),
            Ctx->getFreeDeviceMemoryMegaBytes(),
            (1000.0f / mpiBoardCastData.sharedDataCPUHost.timeAverage));
        glfwSetWindowTitle(window, title);
        mpiBoardCastData.sharedDataCPUHost.elapsedTimeSinceSecond = 0;
    }

    glfwSwapInterval(0);
    glfwSwapBuffers(window);
    glfwPollEvents();
}

Master::Master(int rank, int worldSize)
    : dPtr(new MasterPrivate(rank, worldSize)) {}

Master::~Master() {
    destroy();
}

bool Master::initialize(const uint3& volumeSize) {
    D(Master);
    d->volumeSize = volumeSize;
    return d->initialize();
}

void Master::destroy() {
    _delete(dPtr);
}

void Master::update(float* data) {
    D(Master);
    d->speedSizeArray.setData(data);
    d->update();
}

void Master::render() {
    D(Master);
    d->render();
}

void MasterPrivate::keyboardEvent(GLFWwindow* window, int key, int scanCode, int action, int mods) {

}

void MasterPrivate::mouseButtonEvent(GLFWwindow* window, int button, int action, int mods) {
    switch (action) {
    case GLFW_PRESS:
        mouseButtonState = button;
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            double xPos, yPos;
            glfwGetCursorPos(window, &xPos, &yPos);
            camera.dragStart(xPos, windowHeight - yPos, windowWidth, windowHeight);
        }
        break;
    case GLFW_RELEASE:
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            camera.dragEnd();
        }
        drag = false;
        mousePos.x = 0;
        mousePos.y = 0;
        mouseButtonState = -1;
        break;
    }
}

void MasterPrivate::mouseMoveEvent(GLFWwindow* window, double xPos, double yPos) {
    if (mouseButtonState == GLFW_MOUSE_BUTTON_LEFT) {
        drag = true;
        mousePosPrev.x = mousePos.x;
        mousePosPrev.y = mousePos.y;
        mousePos.x = xPos;
        mousePos.y = windowHeight - yPos;
    }

    if (mouseButtonState == GLFW_MOUSE_BUTTON_RIGHT) {
        camera.dragMove(xPos, windowHeight - yPos, windowWidth, windowHeight);
    }
}

void MasterPrivate::mouseWheelEvent(GLFWwindow* window, double xOffset, double yOffset) {
    if (yOffset < 0) {
        zoom = zoom * 1.1f;
    } else {
        zoom = zoom / 1.1f;
    }
}

void MasterPrivate::resizeEvent(GLFWwindow* window, int width, int height) {
    std::cout << "[MASTER]: Resizing window to (" << width << ", " << height << ")" << std::endl;
    glViewport(0, 0, width, height);
}

MasterPrivate* MasterPrivate::instance;
void MasterPrivate::keyboardEventClb(GLFWwindow* window, int key, int scanCode, int action, int mods) {
    instance->keyboardEvent(window, key, scanCode, action, mods);
}

void MasterPrivate::mouseButtonEventClb(GLFWwindow* window, int button, int action, int mods) {
    instance->mouseButtonEvent(window, button, action, mods);
}

void MasterPrivate::mouseMoveEventClb(GLFWwindow* window, double xPos, double yPos) {
    instance->mouseMoveEvent(window, xPos, yPos);
}

void MasterPrivate::mouseWheelEventClb(GLFWwindow* window, double xOffset, double yOffset) {
    instance->mouseWheelEvent(window, xOffset, yOffset);
}

void MasterPrivate::resizeEventClb(GLFWwindow* window, int width, int height) {
    instance->resizeEvent(window, width, height);
}
