#include "Allocator.hpp"
#include "Auxiliary.hpp"
#include "OpenGLInterop.hpp"
#include "MacCormack.hpp"

#include <memory>

#define DIM 200u
const unsigned int dimX = DIM;
const unsigned int dimY = DIM;
const unsigned int dimZ = DIM;

struct tmp {
    float dim[4];

    int width;
    int height;
    int viewSlice;
    int viewOrientation;

    float mouse[4];
    float dragDirection[4];

    float orientation[4*4]; // replace with glm

    float zoom;
    int smoky;
};

void createVolumes() {
    size_t mem = dimX * dimY * dimZ * 4;
    unsigned short* volume = new unsigned short[mem];
    memset(volume, 0, mem);
}

int main(int argc, char** argv) {
    bool useGpu = true;
    initializeCuda(useGpu);
}