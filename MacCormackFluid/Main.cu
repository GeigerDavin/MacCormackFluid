#include "StdAfx.hpp"
#include "Master.hpp"
#include "Worker.hpp"

#include "Utils/Timer.hpp"

#include "MPI/SharedMatrixZPlanes.hpp"

#define DIMXYZ 100
const int dimX = DIMXYZ;
const int dimY = DIMXYZ;
const int dimZ = DIMXYZ;
int viewSclice = DIMXYZ / 2;
int viewOrientation = 0;

float elapsedTime = 0;
Utils::Timer* timer = nullptr;

int worldSize = -1;
int worldRank = -1;

CUDA::CudaDeviceContext* context;
Master* master = nullptr;
Worker* worker = nullptr;

MPI_BROADCASTDATA mpiBoardCastData;

SharedMatrix<float>* sharedMatrix = nullptr;

bool initialize();
int cleanup();

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    sharedMatrix = new SharedMatrix_ZPlanes<float>(worldRank, worldSize, dimX, dimY, dimZ);

    mpiBoardCastData.sharedDataCPUHost.running = true;
    if (!initialize()) {
    	mpiBoardCastData.sharedDataCPUHost.running = false;
    }

    while (mpiBoardCastData.sharedDataCPUHost.running) {
        Utils::TimerSDK::startTimer(&timer);
        if (worldRank == MPI_MASTER) {
            master->update(sharedMatrix->getMasterMatrix());
            master->render();
        }
        MPI_Bcast(&mpiBoardCastData, sizeof(MPI_BROADCASTDATA), MPI_BYTE, MPI_MASTER, MPI_COMM_WORLD);
        if (worldRank != MPI_MASTER || worldSize == 1) {
            worker->update();
            if(worldSize == 1)
            {
            	worker->getData(sharedMatrix->getMasterMatrixPointer());
            }
            else
            {
            	worker->getData(sharedMatrix->getWorkerMatrixPointer());
            }
        }


        mpiBoardCastData.sharedDataCPUHost.timeDiff = Utils::TimerSDK::getTimerValue(&timer);
        mpiBoardCastData.sharedDataCPUHost.totalTime = Utils::TimerSDK::getTotalTimerValue(&timer);
        mpiBoardCastData.sharedDataCPUHost.timeAverage = Utils::TimerSDK::getAverageTimerValue(&timer);

        sharedMatrix->startMergeMasterMatrixAsync();
        sharedMatrix->waitMergeMasterMatrixAsyncFinish();

        MPI_Barrier(MPI_COMM_WORLD);

        Utils::TimerSDK::stopTimer(&timer);
    }

    return cleanup();
}

bool initialize() {
	context = new CUDA::CudaDeviceContext;
    context->create();
    if (!context->isCreated()) {
        std::cerr << "CPU only not supported" << std::endl;
        return false;
    } else {
        if (worldRank == 0) {
            std::cout << "[MASTER]: CUDA successfully initialized" << std::endl;
        } else if (worldRank == 1) {
            std::cout << "[WORKER]: CUDA successfully initialized" << std::endl;
        }
    }

    if (Utils::TimerSDK::createTimer(&timer)) {
        if (worldRank == 0) {
            std::cout << "[MASTER]: Timer successfully created" << std::endl;
        } else if (worldRank == 1) {
            std::cout << "[WORKER]: Timer successfully created" << std::endl;
        }
    } else {
        return false;
    }

    std::cout << std::endl;

    if (worldRank == MPI_MASTER)  {
        master = new Master(worldRank, worldSize);
        if (!master->initialize(make_uint3(sharedMatrix->getMasterMatX(), sharedMatrix->getMasterMatY(), sharedMatrix->getMasterMatZ()))) {
            return false;
        }
    }
    if (worldRank != MPI_MASTER || worldSize == 1) {
        worker = new Worker(worldRank, worldSize);
        if(worldSize == 1)
        {
            if (!worker->initialize(make_uint3(sharedMatrix->getMasterMatX(), sharedMatrix->getMasterMatY(), sharedMatrix->getMasterMatZ())))
            {
                return false;
            }
        }
        else
        {
			if (!worker->initialize(make_uint3(
					sharedMatrix->getWorkerMatX() - (sharedMatrix->haveGhostCellsXLeft() ? 1 : 0) - (sharedMatrix->haveGhostCellsXRight() ? 1 : 0),
					sharedMatrix->getWorkerMatY() - (sharedMatrix->haveGhostCellsYLeft() ? 1 : 0) - (sharedMatrix->haveGhostCellsYRight() ? 1 : 0),
					sharedMatrix->getWorkerMatZ() - (sharedMatrix->haveGhostCellsZLeft() ? 1 : 0) - (sharedMatrix->haveGhostCellsZRight() ? 1 : 0))))
			{
				return false;
			}
        }
    }

    return true;
}

int cleanup() {
    if (worldRank == 0) {
        std::cout << "[MASTER]: Destroying timer..." << std::endl;
    } else if (worldRank == 1) {
        std::cout << "[WORKER]: Destroying timer..." << std::endl;
    }
    if (Utils::TimerSDK::destroyTimer(&timer)) {
        if (worldRank == 0) {
            std::cout << "[MASTER]: Timer successfully destroyed" << std::endl << std::endl;
        } else if (worldRank == 1) {
            std::cout << "[WORKER]: Timer successfully destroyed" << std::endl << std::endl;
        }
    } else {
        if (worldRank == 0) {
            std::cout << "[MASTER]: Could not destroy timer" << std::endl << std::endl;
        } else if (worldRank == 1) {
            std::cout << "[WORKER]: Could not destroy timer" << std::endl << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (worldRank != MPI_MASTER || worldSize == 1) {
        _delete(worker);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (worldRank == MPI_MASTER) {
        _delete(master);
    }

    _delete(sharedMatrix);

    _delete(context);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
