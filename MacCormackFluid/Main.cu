#include "StdAfx.hpp"

#include "Camera/Camera.hpp"

#include "CUDA/CudaEventTimer.hpp"
#include "CUDA/CudaArray3D.hpp"
#include "CUDA/CudaTextureObject.hpp"
#include "CUDA/CudaSurfaceObject.hpp"
#include "CUDA/CudaGLGraphicsResource.hpp"
#include "CUDA/Modules/DeviceManagement.hpp"

#include "OpenGL/OpenGLBuffer.hpp"
#include "OpenGL/OpenGLShaderProgram.hpp"
#include "OpenGL/OpenGLVertexArrayObject.hpp"

#include "Utils/TrackSphere.hpp"

#include "Kernel/Advect.hpp"
#include "Kernel/Divergence.hpp"
#include "Kernel/Jacobi.hpp"
#include "Kernel/Project.hpp"
#include "Kernel/Render.hpp"


#include <chrono>
#include "MPI/SharedMatrixZPlanes.hpp"

	        //const float4* test = speedArray[0].getHostData();
	        //for (uint i = 0; i < 10; i++) {
	        //    if (test) {
	        //        std::cout << test[i].x << " " << test[i].y << " " << test[i].z << " " << test[i].w << std::endl;
	        //    }
	        //}

#define WINDOW_WIDTH 1280
#define WINDOW_HEIGHT 720
int width = WINDOW_WIDTH;
int height = WINDOW_HEIGHT;
GLFWwindow* mainWindow = nullptr;

#define DIMXYZ 100
const int dimX = DIMXYZ;
const int dimY = DIMXYZ;
const int dimZ = DIMXYZ;

const float sphereRadius = 3.0f;

int viewOrientation = 0;
int viewSclice = DIMXYZ / 2;

int mouseButtonState = -1;
double mousePosX = 0;
double mousePosY = 0;

double mousePosXPrev = 0;
double mousePosYPrev = 0;

bool drag = false;
int mousePosXDragStart = 0;
int mousePosYDragStart = 0;
int mousePosXDragEnd = 0;
int mousePosYDragEnd = 0;
float zoom = 1.2f;

Utils::TrackSphere trackSphere;

float elapsedTime = 0;
Utils::Timer* timer = nullptr;

////////////////////////////////////////////////////////////////////////////////
// CUDA Arrays pointing to the GPU residing data, must be accessed by textures and surfaces
////////////////////////////////////////////////////////////////////////////////
CUDA::CudaArray3D<float4> speedArray[3];
CUDA::CudaArray3D<float> speedSizeArrayWorker;
CUDA::CudaArray3D<float4> pressureArray[2];
CUDA::CudaArray3D<float4> divergenceArray;

////////////////////////////////////////////////////////////////////////////////
// Surfaces for writing to and reading from without linear interpolation and multisampling
////////////////////////////////////////////////////////////////////////////////
CUDA::CudaSurfaceObject speedSurface[3];
CUDA::CudaSurfaceObject speedSizeSurfaceWorker;
CUDA::CudaSurfaceObject pressureSurface[2];
CUDA::CudaSurfaceObject divergenceSurface;

////////////////////////////////////////////////////////////////////////////////
// Read-only textures for reading from with linear interpolation and multisampling
////////////////////////////////////////////////////////////////////////////////
CUDA::CudaTextureObject speedTexture[3];
CUDA::CudaTextureObject speedSizeTextureWorker;
CUDA::CudaTextureObject pressureTexture[2];
CUDA::CudaTextureObject divergenceTexture;

////////////////////////////////////////////////////////////////////////////////
// View output
////////////////////////////////////////////////////////////////////////////////
//#if USE_TEXTURE_2D
// GLuint viewGLTexture;
// cudaGraphicsResource_t viewGraphicsResource;
//#else

CUDA::CudaArray3D<float> speedSizeArrayMaster;
CUDA::CudaSurfaceObject speedSizeSurfaceMaster;
CUDA::CudaTextureObject speedSizeTextureMaster;

SharedMatrix<float>* speedSizeOutput = nullptr;
 OpenGL::OpenGLBuffer rgbaBuffer;
 CUDA::CudaGLGraphicsResource rgbaGraphicsResource;
//#endif

////////////////////////////////////////////////////////////////////////////////
// MPI
////////////////////////////////////////////////////////////////////////////////
int world_size;
int world_rank;

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////
void master();
void slave();

bool initializeGL();
bool createVolumes();
bool createSpeedSizeVolumeWorker();
bool createSpeedSizeVolumeMaster();
bool createViewOutput();

void updateConstants();
void updateVolumes();
void renderVolumes();

void cleanup();

static void cursorPosClb(GLFWwindow* window, double xpos, double ypos);
static void cursorButtonClb(GLFWwindow* window, int button, int action, int mods);
static void cursorScrollClb(GLFWwindow* window, double xoffset, double yoffset);
static void windowResizeClb(GLFWwindow* window, int width, int height);

////////////////////////////////////////////////////////////////////////////////
// Definitions
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char *argv[]) {
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	hostConstant.running = true;

	speedSizeOutput = new SharedMatrix_ZPlanes<float>(world_rank, world_size, dimX, dimY, dimZ);
	std::cout << "Worker Size " << world_rank << ": " << speedSizeOutput->getWorkerMatX() << " " << speedSizeOutput->getWorkerMatY() << " " << speedSizeOutput->getWorkerMatZ() << std::endl;

    CUDA::DeviceManagement::initializeCuda(true);
    if (!CUDA::useCuda) {
        std::cerr << "CPU only not supported" << std::endl;
        cleanup();
        return false;
    } else {
    	std::cerr << "Worker " << world_rank << ": " << "CUDA initialized" << std::endl;
    }

    if (Utils::TimerSDK::createTimer(&timer)) {
        std::cout << "Timer successfully created" << std::endl;
    } else {
        cleanup();
        return false;
    }

	if (world_rank == MPI_MASTER) {
		master();
	} else {
		slave();
	}

	if(speedSizeOutput)
		delete speedSizeOutput;

    cleanup();
    MPI_Finalize();
    return EXIT_SUCCESS;
}

void master() {
    if (!createSpeedSizeVolumeMaster()) {
    	cleanup();
    	return;
    }
	if (!initializeGL()) {
		cleanup();
		return;
	}
    if (!createViewOutput()) {
    	cleanup();
    	return;
    }

    MPI_Status status;
	while (hostConstant.running) {
		if (glfwGetKey(mainWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
			glfwSetWindowShouldClose(mainWindow, true);
			hostConstant.running = false;
		}

		std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
		updateConstants();
		MPI_Bcast(&hostConstant, sizeof(Constant), MPI_BYTE, MPI_MASTER, MPI_COMM_WORLD);
		CUDA::MemoryManagement::moveHostToSymbol(deviceConstant, hostConstant);

		speedSizeOutput->startMergeMasterMatrixAsync();
		speedSizeOutput->waitMergeMasterMatrixAsyncFinish();

		renderVolumes();

		glfwSwapInterval(0);
		glfwSwapBuffers(mainWindow);
		glfwPollEvents();

		std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
		//float dt = Utils::TimerSDK::getTimerValue(&timer);
		float dt = (float) duration;
		elapsedTime += dt;
		if (true) {
			char title[256];
			sprintf(title, "MacCormack      Delta Time: %.2fms       Average Time: %.2fms        Total Time: %.0fms       %.0fFPS",
				(float) duration,
				Utils::TimerSDK::getAverageTimerValue(&timer),
				Utils::TimerSDK::getTotalTimerValue(&timer),
				(float) (elapsedTime / dt));
			glfwSetWindowTitle(mainWindow, title);
			elapsedTime = 0;
		}

		//MPI_Barrier(MPI_COMM_WORLD);
	}
}

void slave() {
    if (!createVolumes()) {
    	cleanup();
    	return;
    }
    if (!createSpeedSizeVolumeWorker()) {
    	cleanup();
    	return;
    }

	while (hostConstant.running) {
		MPI_Bcast(&hostConstant, sizeof(Constant), MPI_BYTE, MPI_MASTER, MPI_COMM_WORLD);
		//printf("%f\n", hostConstant.mouse.x);
		CUDA::MemoryManagement::moveHostToSymbol(deviceConstant, hostConstant);

		updateVolumes();
		speedSizeOutput->startMergeMasterMatrixAsync();
		//speedSizeOutput->startExchangeGhostCells();
		speedSizeOutput->waitMergeMasterMatrixAsyncFinish();
		//speedSizeOutput->waitExchangeGhostCellsFinish();
		//MPI_Barrier(MPI_COMM_WORLD);
	}
}

namespace {
    void glfwErrorCallback(int error, const char* description) {
        std::cerr << "GLFW Error " << error << ": " << description << std::endl;
        cleanup();
        exit(EXIT_FAILURE);
    }
}

bool initializeGL() {
    glfwSetErrorCallback(glfwErrorCallback);

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW. Error = " << glGetError() << std::endl;
        return false;
    }

    mainWindow = glfwCreateWindow(width, height, "MacCormack", NULL, NULL);
    if (!mainWindow) {
        std::cerr << "Failed to create glWindow. Error = " << glGetError() << std::endl;
        return false;
    }
    glfwMakeContextCurrent(mainWindow);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW. Error = " << glGetError() << std::endl;
        return false;
    } else {
        printf("_________________________________________\n");
        std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
        std::cout << "GLEW version: " << glewGetString(GLEW_VERSION) << std::endl;
        printf("_________________________________________\n");
        printf("\n");
    }

    glfwSetCursorPosCallback(mainWindow, cursorPosClb);
    glfwSetMouseButtonCallback(mainWindow, cursorButtonClb);
    glfwSetScrollCallback(mainWindow, cursorScrollClb);
    glfwSetWindowSizeCallback(mainWindow, windowResizeClb);

    return true;
}

bool createVolumes() {
    // Speed (vectorial)
    for (int i = 0; i < 3; i++) {
        speedArray[i].create(speedSizeOutput->getWorkerMatX(), speedSizeOutput->getWorkerMatY(), speedSizeOutput->getWorkerMatZ());
        speedTexture[i].setFilterMode(CUDA::CudaTextureObject::FilterMode::LinearFilter);
        speedTexture[i].setNormalized(true);
        if (speedTexture[i].create(speedArray[i].get())) {
            std::cout << "Speed texture object " << speedTexture[i].getId() << " successfully created" << std::endl;
        } else {
            return false;
        }
        if (speedSurface[i].create(speedArray[i].get())) {
            std::cout << "Speed surface object " << speedSurface[i].getId() << " successfully created" << std::endl;
        } else {
            return false;
        }
    }

    // Pressure (vectorial)
    for (int i = 0; i < 2; i++) {
        pressureArray[i].create(speedSizeOutput->getWorkerMatX(), speedSizeOutput->getWorkerMatY(), speedSizeOutput->getWorkerMatZ());
        pressureTexture[i].setFilterMode(CUDA::CudaTextureObject::FilterMode::LinearFilter);
        pressureTexture[i].setNormalized(true);
        if (pressureTexture[i].create(pressureArray[i].get())) {
            std::cout << "Pressure texture object " << pressureTexture[i].getId() << " successfully created" << std::endl;
        } else {
            return false;
        }
        if (pressureSurface[i].create(pressureArray[i].get())) {
            std::cout << "Pressure surface object " << pressureSurface[i].getId() << " successfully created" << std::endl;
        } else {
            return false;
        }
    }

    // Divergence (vectorial)
    divergenceArray.create(speedSizeOutput->getWorkerMatX(), speedSizeOutput->getWorkerMatY(), speedSizeOutput->getWorkerMatZ());
    divergenceTexture.setFilterMode(CUDA::CudaTextureObject::FilterMode::LinearFilter);
    divergenceTexture.setNormalized(true);
    if (divergenceTexture.create(divergenceArray.get())) {
        std::cout << "Divergence texture object " << divergenceTexture.getId() << " successfully created" << std::endl;
    } else {
        return false;
    }
    if (divergenceSurface.create(divergenceArray.get())) {
        std::cout << "Divergence surface object " << divergenceSurface.getId() << " successfully created" << std::endl;
    } else {
        return false;
    }

    std::cout << std::endl;
    return true;
}

bool createSpeedSizeVolumeWorker() {
    // Speed size (scalar)
    speedSizeArrayWorker.create(speedSizeOutput->getWorkerMatX(), speedSizeOutput->getWorkerMatY(), speedSizeOutput->getWorkerMatZ());
    speedSizeTextureWorker.setFilterMode(CUDA::CudaTextureObject::FilterMode::LinearFilter);
    speedSizeTextureWorker.setNormalized(true);
    if (speedSizeTextureWorker.create(speedSizeArrayWorker.get())) {
        std::cout << "Speed size texture object worker " << speedSizeTextureWorker.getId() << " successfully created" << std::endl;
    } else {
        return false;
    }
    if (speedSizeSurfaceWorker.create(speedSizeArrayWorker.get())) {
        std::cout << "Speed size surface object worker " << speedSizeSurfaceWorker.getId() << " successfully created" << std::endl;
    } else {
        return false;
    }

    std::cout << std::endl;
    return true;
}

bool createSpeedSizeVolumeMaster() {
    // Speed size (scalar)
    speedSizeArrayMaster.create(speedSizeOutput->getMasterMatX(), speedSizeOutput->getMasterMatY(), speedSizeOutput->getMasterMatZ());
    speedSizeTextureMaster.setFilterMode(CUDA::CudaTextureObject::FilterMode::LinearFilter);
    speedSizeTextureMaster.setNormalized(true);
    if (speedSizeTextureMaster.create(speedSizeArrayMaster.get())) {
        std::cout << "Speed size texture object master " << speedSizeTextureMaster.getId() << " successfully created" << std::endl;
    } else {
        return false;
    }
    if (speedSizeSurfaceMaster.create(speedSizeArrayMaster.get())) {
        std::cout << "Speed size surface object master " << speedSizeSurfaceMaster.getId() << " successfully created" << std::endl;
    } else {
        return false;
    }

    std::cout << std::endl;
    return true;
}

bool createViewOutput() {
#if USE_TEXTURE_2D
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &viewGLTexture);
    glBindTexture(GL_TEXTURE_2D, viewGLTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);

    checkCudaError(cudaGraphicsGLRegisterImage(&viewGraphicsResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
#else
    rgbaBuffer.setType(OpenGL::OpenGLBuffer::Type::PixelUnpackBuffer);
    rgbaBuffer.setUsagePattern(OpenGL::OpenGLBuffer::UsagePattern::StreamDraw);
    rgbaBuffer.create();
    rgbaBuffer.bind();
    rgbaBuffer.allocate(width * height * sizeof(GLubyte) * 4);
    rgbaBuffer.release();

    rgbaGraphicsResource.setRegisterFlag(CUDA::CudaGLGraphicsResource::RegisterFlag::WriteDiscard);
    rgbaGraphicsResource.create(rgbaBuffer.getId());
#endif

    return true;
}

void updateConstants() {
    hostConstant.volumeSize.x = dimX;
    hostConstant.volumeSize.y = dimY;
    hostConstant.volumeSize.z = dimZ;
    hostConstant.volumeSize.w = 0;

    hostConstant.viewPort.x = width;
    hostConstant.viewPort.y = height;

    hostConstant.viewSlice = viewSclice;
    hostConstant.viewOrientation = viewOrientation;

    hostConstant.mouse.x = (float) mousePosX;
    hostConstant.mouse.y = (float) mousePosY;
    hostConstant.mouse.z = viewSclice;
    hostConstant.mouse.w = sphereRadius;

    if (drag) {
        hostConstant.dragDirection.x = (float) (mousePosX - mousePosXPrev) * dimX / width;
        hostConstant.dragDirection.y = (float) (mousePosY - mousePosYPrev) * dimY / height;
    } else {
        hostConstant.dragDirection.x = 0;
        hostConstant.dragDirection.y = 0;
    }
    hostConstant.dragDirection.z = 0;

    Utils::Mat3 rot = trackSphere.getRotationMatrix();
    for (int i = 0; i < 3; i++) {
        hostConstant.rotation.m[i].x = rot(i, 0);
        hostConstant.rotation.m[i].y = rot(i, 1);
        hostConstant.rotation.m[i].z = rot(i, 2);
    }
    hostConstant.zoom = zoom;
}

template <class Resource>
void swap(Resource& res1, Resource& res2) {
    Resource res = std::move(res1);
    res1 = std::move(res2);
    res2 = std::move(res);
}

void updateVolumes() {
    //checkCudaError(cudaGraphicsMapResources(1, &viewGraphicsResource));
    //cudaArray_t viewCudaArray;
    //checkCudaError(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewGraphicsResource, 0, 0));
    //cudaResourceDesc viewCudaArrayResourceDesc;
    //viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
    //viewCudaArrayResourceDesc.res.array.array = viewCudaArray;
    //cudaSurfaceObject_t viewCudaSurfaceObject;
    //checkCudaError(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));

    Utils::TimerSDK::startTimer(&timer);

	int x = speedSizeOutput->getWorkerMatX();
	int y = speedSizeOutput->getWorkerMatY();
	int z = speedSizeOutput->getWorkerMatZ() / 2;

    dim3 blockSize(16, 4, 4);
    dim3 gridSize((x + 15) / 16, y / 4, z / 4);

    Kernel::advect3D<<<gridSize, blockSize>>>(speedTexture[0].getId(),              // Input 0
                                              speedSurface[0].getId(),              // Input 0
                                              speedSurface[1].getId());             // Output 1
    getLastCudaError("advect3D kernel failed");
    //CUDA::DeviceManagement::deviceSync();

    Kernel::advectBackward3D<<<gridSize, blockSize>>>(speedTexture[1].getId(),      // Input 1
                                                      speedSurface[0].getId(),      // Input 0
                                                      speedSurface[2].getId());     // Output 2
    getLastCudaError("advectBackward3D kernel failed");
    //CUDA::DeviceManagement::deviceSync();

    Kernel::advectMacCormack3D<<<gridSize, blockSize>>>(speedTexture[0].getId(),    // Input 0
                                                        speedSurface[0].getId(),    // Input 0
                                                        speedTexture[2].getId(),    // Input 2
                                                        speedSurface[1].getId());   // Output 1
    getLastCudaError("advectMacCormack3D kernel failed");
    //CUDA::DeviceManagement::deviceSync();

    Kernel::renderSphere<<<gridSize, blockSize>>>(speedSurface[1].getId(),          // Input 1
                                                  speedSurface[0].getId());         // Output 0
    getLastCudaError("renderSphere kernel failed");
    //CUDA::DeviceManagement::deviceSync();

    gridSize = dim3((x + 63) / 64, y / 4, z / 4);
    Kernel::divergence3D<<<gridSize, blockSize>>>(speedSurface[0].getId(),          // Input 0
                                                  divergenceSurface.getId());       // Output
    getLastCudaError("divergence3D kernel failed");
    //CUDA::DeviceManagement::deviceSync();

    gridSize = dim3((x + 63) / 64, y / 4, z / 4);
    for (int i = 0; i < 10; i++) {
        Kernel::jacobi3D<<<gridSize, blockSize>>>(pressureSurface[1].getId(),
                                                  divergenceSurface.getId(),
                                                  pressureSurface[0].getId());
        getLastCudaError("jacobi3D kernel failed");
        //CUDA::DeviceManagement::deviceSync();

        Kernel::jacobi3D<<<gridSize, blockSize>>>(pressureSurface[0].getId(),
                                                  divergenceSurface.getId(),
                                                  pressureSurface[1].getId());
        getLastCudaError("jacobi3D kernel failed");
        //CUDA::DeviceManagement::deviceSync();
    }

    gridSize = dim3((x + 63) / 64, y / 4, z / 4);
    Kernel::project3D<<<gridSize, blockSize>>>(speedSurface[0].getId(),             // Input 0
                                               pressureSurface[1].getId(),          // Input 1
                                               speedSurface[1].getId(),             // Output 1
                                               speedSizeSurfaceWorker.getId());           // Output
    getLastCudaError("project3D kernel failed");
    //CUDA::DeviceManagement::deviceSync();

    //checkCudaError(cudaDestroySurfaceObject(viewCudaSurfaceObject));
    //checkCudaError(cudaGraphicsUnmapResources(1, &viewGraphicsResource));

    speedSizeArrayWorker.getHostData(&(speedSizeOutput->_workerMatrix));

    swap(speedTexture[0], speedTexture[1]);
    swap(speedSurface[0], speedSurface[1]);

    Utils::TimerSDK::stopTimer(&timer);
    std::cout << "Worker " << world_rank << ": " << Utils::TimerSDK::getTimerValue(&timer) << std::endl;
}

void renderVolumes() {
	//Utils::TimerSDK::startTimer(&timer);

	speedSizeArrayMaster.setData(speedSizeOutput->_masterMatrix);

    uint* rgba = (uint *)rgbaGraphicsResource.map();
    CUDA::MemoryManagement::deviceMemset(rgba, 0, width * height * 4);

    dim3 blockSize = dim3(16, 16, 1);
    dim3 gridSize = dim3((width + 15) / 16, (height + 15) / 16, 1);
    Kernel::renderVolume<<<gridSize, blockSize>>>(speedSizeTextureMaster.getId(),  // Input
                                                  rgba);               // Output
    getLastCudaError("renderVolume kernel failed");
    //CUDA::DeviceManagement::deviceSync();

    rgbaGraphicsResource.unmap();

    //Utils::TimerSDK::stopTimer(&timer);
    //std::cout << "Worker " << world_rank << ": " << Utils::TimerSDK::getTimerValue(&timer) << std::endl;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

#if 1
    rgbaBuffer.bind();
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    rgbaBuffer.release();
#else
    rgbaBuffer.bind();
    glBindTexture(GL_TEXTURE_2D, rgbaTex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    rgbaBuffer.release();

    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glBindTexture(GL_TEXTURE_2D, viewGLTexture);
    glBegin(GL_QUADS);
    {
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f(+1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f(+1.0f, +1.0f);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, +1.0f);
    }
    glEnd();
    glBindTexture(GL_TEXTURE_2D, 0);
    glFinish();
#endif
}

void cleanup() {
    std::cout << "Destroying textures..." << std::endl;
    for (int i = 0; i < 3; i++) {
        speedTexture[i].destroy();
    }
    speedSizeTextureWorker.destroy();
    speedSizeTextureMaster.destroy();
    for (int i = 0; i < 2; i++) {
        pressureTexture[i].destroy();
    }
    divergenceTexture.destroy();
    std::cout << "Textures successfully destroyed" << std::endl << std::endl;

    std::cout << "Destroying surfaces..." << std::endl;
    for (int i = 0; i < 3; i++) {
        speedSurface[i].destroy();
    }
    speedSizeSurfaceWorker.destroy();
    speedSizeSurfaceMaster.destroy();
    for (int i = 0; i < 2; i++) {
        pressureSurface[i].destroy();
    }
    divergenceSurface.destroy();
    std::cout << "Surfaces successfully destroyed" << std::endl << std::endl;

    std::cout << "Destroying CUDA arrays..." << std::endl;
    for (int i = 0; i < 3; i++) {
        speedArray[i].destroy();
    }
    speedSizeArrayWorker.destroy();
    speedSizeArrayMaster.destroy();
    for (int i = 0; i < 2; i++) {
        pressureArray[i].destroy();
    }
    divergenceArray.destroy();
    std::cout << "CUDA arrays successfully destroyed" << std::endl << std::endl;

    std::cout << "Destroying view output..." << std::endl;
    rgbaBuffer.destroy();
    rgbaGraphicsResource.destroy();
    //checkCudaError(cudaGraphicsUnregisterResource(viewGraphicsResource));
    //glDeleteTextures(1, &viewGLTexture);
    std::cout << "View output successfully destroyed" << std::endl << std::endl;

    if (mainWindow) {
        std::cout << "Destroying main window..." << std::endl;
        glfwDestroyWindow(mainWindow);
        mainWindow = nullptr;
        std::cout << "Main window successfully destroyed" << std::endl << std::endl;
    }
    glfwTerminate();

    std::cout << "Destroying timer..." << std::endl;
    if (Utils::TimerSDK::destroyTimer(&timer)) {
        std::cout << "Timer sucessfully destroyed" << std::endl << std::endl;
    }
    CUDA::DeviceManagement::deviceReset();
}

static void cursorPosClb(GLFWwindow* window, double xpos, double ypos) {
    mousePosXPrev = mousePosX;
    mousePosYPrev = mousePosY;

    if (mouseButtonState == GLFW_MOUSE_BUTTON_LEFT) {
        mousePosX = xpos;
        mousePosY = height - ypos;
    }

    if (mouseButtonState == GLFW_MOUSE_BUTTON_RIGHT) {
        trackSphere.dragMove(xpos, height - ypos, width, height);
    }
}

static void cursorButtonClb(GLFWwindow* window, int button, int action, int mods) {
    switch (action) {
    case GLFW_PRESS:
        mouseButtonState = button;
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            double xPos, yPos;
            glfwGetCursorPos(window, &xPos, &yPos);
            trackSphere.dragStart(xPos, height - yPos, width, height);
        }
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            drag = true;
            double xPos, yPos;
            glfwGetCursorPos(window, &xPos, &yPos);
            mousePosX = xPos;
            mousePosY = yPos;
        }
        break;
    case GLFW_RELEASE:
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            trackSphere.dragEnd();
        }
        drag = false;
        mousePosX = 0;
        mousePosY = 0;
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

static void windowResizeClb(GLFWwindow* window, int width, int height) {
    std::cout << "Resizing window to (" << width << ", " << height << ")" << std::endl;
    glViewport(0, 0, width, height);
}
