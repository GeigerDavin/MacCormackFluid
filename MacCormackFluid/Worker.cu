#include "StdAfx.hpp"
#include "Worker.hpp"

#include "Kernel/Advect.hpp"
#include "Kernel/Divergence.hpp"
#include "Kernel/Jacobi.hpp"
#include "Kernel/Project.hpp"
#include "Kernel/Render.hpp"

class WorkerPrivate {
public:
    WorkerPrivate(int rank, int worldSize)
        : mpiRank(rank)
        , mpiWorldSize(worldSize) {

        memset(&volumeSize, 0, sizeof(uint3));
    }

    ~WorkerPrivate() {
        destroy();
    }

    bool initialize();
    void destroy();
    void update();

public:
    int mpiRank;
    int mpiWorldSize;

    uint3 volumeSize;

    ////////////////////////////////////////////////////////////////////////////////
    // CUDA Arrays pointing to the GPU residing data, must be accessed by textures and surfaces
    ////////////////////////////////////////////////////////////////////////////////
    CUDA::CudaArray3D<float4> speedArray[3];
    CUDA::CudaArray3D<float> speedSizeArray;
    CUDA::CudaArray3D<float4> pressureArray[2];
    CUDA::CudaArray3D<float4> divergenceArray;

    ////////////////////////////////////////////////////////////////////////////////
    // Surfaces for writing to and reading from without linear interpolation and multisampling
    ////////////////////////////////////////////////////////////////////////////////
    CUDA::CudaSurfaceObject speedSurface[3];
    CUDA::CudaSurfaceObject speedSizeSurface;
    CUDA::CudaSurfaceObject pressureSurface[2];
    CUDA::CudaSurfaceObject divergenceSurface;

    ////////////////////////////////////////////////////////////////////////////////
    // Read-only textures for reading from with linear interpolation and multisampling
    ////////////////////////////////////////////////////////////////////////////////
    CUDA::CudaTextureObject speedTexture[3];
    CUDA::CudaTextureObject speedSizeTexture;
    CUDA::CudaTextureObject pressureTexture[2];
    CUDA::CudaTextureObject divergenceTexture;
};

bool WorkerPrivate::initialize() {
    for (int i = 0; i < 3; i++) {
        speedArray[i].create(volumeSize.x, volumeSize.y, volumeSize.z);
        speedTexture[i].setFilterMode(CUDA::CudaTextureObject::FilterMode::LinearFilter);
        speedTexture[i].setNormalized(true);
        if (speedTexture[i].create(speedArray[i].get())) {
            std::cout << "[WORKER]: Speed texture object " << speedTexture[i].getId() << " successfully created" << std::endl;
        } else {
            return false;
        }
        if (speedSurface[i].create(speedArray[i].get())) {
            std::cout << "[WORKER]: Speed surface object " << speedSurface[i].getId() << " successfully created" << std::endl;
        } else {
            return false;
        }
    }

    for (int i = 0; i < 2; i++) {
        pressureArray[i].create(volumeSize.x, volumeSize.y, volumeSize.z);
        pressureTexture[i].setFilterMode(CUDA::CudaTextureObject::FilterMode::LinearFilter);
        pressureTexture[i].setNormalized(true);
        if (pressureTexture[i].create(pressureArray[i].get())) {
            std::cout << "[WORKER]: Pressure texture object " << pressureTexture[i].getId() << " successfully created" << std::endl;
        } else {
            return false;
        }
        if (pressureSurface[i].create(pressureArray[i].get())) {
            std::cout << "[WORKER]: Pressure surface object " << pressureSurface[i].getId() << " successfully created" << std::endl;
        } else {
            return false;
        }
    }

    divergenceArray.create(volumeSize.x, volumeSize.y, volumeSize.z);
    divergenceTexture.setFilterMode(CUDA::CudaTextureObject::FilterMode::LinearFilter);
    divergenceTexture.setNormalized(true);
    if (divergenceTexture.create(divergenceArray.get())) {
        std::cout << "[WORKER]: Divergence texture object " << divergenceTexture.getId() << " successfully created" << std::endl;
    } else {
        return false;
    }
    if (divergenceSurface.create(divergenceArray.get())) {
        std::cout << "[WORKER]: Divergence surface object " << divergenceSurface.getId() << " successfully created" << std::endl;
    } else {
        return false;
    }

    speedSizeArray.create(volumeSize.x, volumeSize.y, volumeSize.z);
    speedSizeTexture.setFilterMode(CUDA::CudaTextureObject::FilterMode::LinearFilter);
    speedSizeTexture.setNormalized(true);
    if (speedSizeTexture.create(speedSizeArray.get())) {
        std::cout << "[WORKER]: Speed size texture object " << speedSizeTexture.getId() << " successfully created" << std::endl;
    } else {
        return false;
    }
    if (speedSizeSurface.create(speedSizeArray.get())) {
        std::cout << "[WORKER]: Speed size surface object " << speedSizeSurface.getId() << " successfully created" << std::endl;
    } else {
        return false;
    }

    float4 volumeSizeHost = make_float4(volumeSize.x, volumeSize.y, volumeSize.z, 0.0f);
    CUDA::MemoryManagement::moveHostToSymbol(volumeSizeDev, volumeSizeHost);

    std::cout << std::endl;
    return true;
}

void WorkerPrivate::destroy() {
    std::cout << "[WORKER]: Destroying textures..." << std::endl;
    for (int i = 0; i < 3; i++) {
        speedTexture[i].destroy();
    }
    for (int i = 0; i < 2; i++) {
        pressureTexture[i].destroy();
    }
    divergenceTexture.destroy();
    speedSizeTexture.destroy();
    std::cout << "[WORKER]: Textures successfully destroyed" << std::endl << std::endl;

    std::cout << "[WORKER]:Destroying surfaces..." << std::endl;
    for (int i = 0; i < 3; i++) {
        speedSurface[i].destroy();
    }
    for (int i = 0; i < 2; i++) {
        pressureSurface[i].destroy();
    }
    divergenceSurface.destroy();
    speedSizeSurface.destroy();
    std::cout << "[WORKER]: Surfaces successfully destroyed" << std::endl << std::endl;

    std::cout << "[WORKER]: Destroying CUDA arrays..." << std::endl;
    for (int i = 0; i < 3; i++) {
        speedArray[i].destroy();
    }
    for (int i = 0; i < 2; i++) {
        pressureArray[i].destroy();
    }
    divergenceArray.destroy();
    speedSizeArray.destroy();
    std::cout << "[WORKER]: CUDA arrays successfully destroyed" << std::endl << std::endl;
}

namespace {
    template <class Resource>
    void swap(Resource& res1, Resource& res2) {
        Resource res = std::move(res1);
        res1 = std::move(res2);
        res2 = std::move(res);
    }
}

void WorkerPrivate::update() {
    CUDA::MemoryManagement::moveHostToSymbol(g, mpiBoardCastData.sharedDataGPUHost);

    dim3 blockSize(16, 4, 4);
    dim3 gridSize((volumeSize.x + 15) / 16, volumeSize.y / 4, volumeSize.z / 4);


    Kernel::advect3D<<<gridSize, blockSize>>>(speedTexture[0].getId(),              // Input speed 0
                                              speedSurface[0].getId(),              // Input speed 0
                                              speedSurface[1].getId());             // Output speed 1
//    getLastCudaError("advect3D kernel failed");
//
    Kernel::advectBackward3D<<<gridSize, blockSize>>>(speedTexture[1].getId(),      // Input speed 1
                                                      speedSurface[0].getId(),      // Input speed 0
                                                      speedSurface[2].getId());     // Output speed 2
//    getLastCudaError("advectBackward3D kernel failed");
//
    Kernel::advectMacCormack3D<<<gridSize, blockSize>>>(speedTexture[0].getId(),    // Input speed 0
                                                        speedSurface[0].getId(),    // Input speed 0
                                                        speedTexture[2].getId(),    // Input speed 2
                                                        speedSurface[1].getId());   // Output speed 1
//    getLastCudaError("advectMacCormack3D kernel failed");
//
    Kernel::renderSphere<<<gridSize, blockSize>>>(speedSurface[1].getId(),          // Input speed 1
                                                  speedSurface[0].getId());         // Output speed 0
//    getLastCudaError("renderSphere kernel failed");
//
    gridSize = dim3((volumeSize.x + 63) / 64, volumeSize.y / 4, volumeSize.z / 4);
    Kernel::divergence3D<<<gridSize, blockSize>>>(speedSurface[0].getId(),          // Input speed 0
                                                  divergenceSurface.getId());       // Output divergence
    //getLastCudaError("divergence3D kernel failed");

    gridSize = dim3((volumeSize.x + 63) / 64, volumeSize.y / 4, volumeSize.z / 4);
    for (int i = 0; i < 10; i++) {
        Kernel::jacobi3D<<<gridSize, blockSize>>>(pressureSurface[1].getId(),       // Input pressure 1
                                                  divergenceSurface.getId(),        // Input divergence
                                                  pressureSurface[0].getId());      // Output pressure 0
        //getLastCudaError("jacobi3D kernel failed");

        Kernel::jacobi3D<<<gridSize, blockSize>>>(pressureSurface[0].getId(),       // Input pressure 0
                                                  divergenceSurface.getId(),        // Input divergence
                                                  pressureSurface[1].getId());      // Output pressure 1
        //getLastCudaError("jacobi3D kernel failed");
    }

    gridSize = dim3((volumeSize.x + 63) / 64, volumeSize.y / 4, volumeSize.z / 4);
    Kernel::project3D<<<gridSize, blockSize>>>(speedSurface[0].getId(),             // Input speed 0
                                               pressureSurface[1].getId(),          // Input pressure 1
                                               speedSurface[1].getId(),             // Output speed 11
                                               speedSizeSurface.getId());           // Output speed size
    //getLastCudaError("project3D kernel failed");
    //Ctx->synchronize();

    swap(speedTexture[0], speedTexture[1]);
    swap(speedSurface[0], speedSurface[1]);
}

Worker::Worker(int rank, int worldSize)
    : dPtr(new WorkerPrivate(rank, worldSize)) {}

Worker::~Worker() {
    destroy();
}

bool Worker::initialize(const uint3& volumeSize) {
    D(Worker);
    d->volumeSize = volumeSize;
    return d->initialize();
}

void Worker::destroy() {
    _delete(dPtr);
}

void Worker::update() {
    D(Worker);
    d->update();
}

void Worker::getData(float** data) const {
	D(const Worker);
	d->speedSizeArray.getHostData(data);
}


