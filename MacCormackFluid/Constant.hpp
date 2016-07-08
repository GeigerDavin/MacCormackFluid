#ifndef CONSTANT_HPP
#define CONSTANT_HPP

typedef struct {
    float4 m[4];
} float4x4;

struct __align__(128) Constant {
    float4 volumeSize;

    uint2 viewPort;
    uint viewSlice;
    uint viewOrientation;

    float4 mouse;
    float4 dragDirection;

    float4x4 rotation;

    float zoom;
    int smoky;

    bool running;
};

static Constant hostConstant;
__constant__ static Constant deviceConstant;

#endif
