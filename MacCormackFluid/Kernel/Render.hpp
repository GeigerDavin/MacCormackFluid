#ifndef RENDER_HPP
#define RENDER_HPP

namespace Kernel {

DINLINE float3 mul(const float4x4& m, const float3& v) {
    float3 r;
    r.x = dot(v, make_float3(m.m[0]));
    r.y = dot(v, make_float3(m.m[1]));
    r.z = dot(v, make_float3(m.m[2]));
    return r;
}

DINLINE float4 mul(const float4x4& m, const float4& v) {
    float4 r;
    r.x = dot(v, m.m[0]);
    r.y = dot(v, m.m[1]);
    r.z = dot(v, m.m[2]);
    r.w = 1.0f;
    return r;
}

DINLINE uint rgbaFloatToInt(float4 rgba) {
    rgba.x = __saturatef(rgba.x);
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

__global__ static void renderSphere(cudaSurfaceObject_t speedIn,
                                    cudaSurfaceObject_t speedOut) {
    TID;
    if (tid.x >= deviceConstant.volumeSize.x) {
        return;
    }

    float3 eye = make_float3(0.0f, 0.0f, 4.0f);
    float testX = 2 * deviceConstant.mouse.x;
    float testY = 2 * deviceConstant.mouse.y;
    testX -= (float) deviceConstant.viewPort.x;
    testY -= (float) deviceConstant.viewPort.y;
    float bla = fminf(deviceConstant.viewPort.x, deviceConstant.viewPort.y);
    testX /= bla;
    testY /= bla;
    testX *= deviceConstant.zoom;
    testY *= deviceConstant.zoom;
    float3 rayDir = make_float3(testX, testY, 0) - eye;

    float3 viewDir = 0 - eye;

    float t = -dot(viewDir, eye) / dot(viewDir, rayDir);

    float3 ball = eye + t * rayDir;

    ball = mul(deviceConstant.rotation, ball);
    float4 force = mul(deviceConstant.rotation, deviceConstant.dragDirection) * deviceConstant.zoom * 4;

    // Texture coordinates of intersection
    ball.x = deviceConstant.volumeSize.x * (ball.x + 1) / 2;
    ball.y = deviceConstant.volumeSize.y * (ball.y + 1) / 2;
    ball.z = deviceConstant.volumeSize.z * (ball.z + 1) / 2;

    // Volume coordinates relative to sphere center
    ball.x = tid.x - ball.x;
    ball.y = tid.y - ball.y;
    ball.z = tid.z - ball.z;

    float r = dot(ball, ball); // Radius^2 square

    // Draw sphere
    float g = exp(-r / (deviceConstant.mouse.w * deviceConstant.mouse.w));
    float4 speed = surf3Dread<float4>(speedIn, tid.x * sizeof(float4), tid.y, tid.z);

    float4 sphere = make_float4(speed.x + force.x * g,
                                speed.y + force.y * g,
                                speed.z + force.z * g,
                                speed.w + length(force) * g);

    surf3Dwrite(sphere, speedOut, tid.x * sizeof(float4), tid.y, tid.z);
}

__global__ static void renderVolume(cudaTextureObject_t speedSizeInTex,
                                    uint* viewOutput)
{
    TID;
    float3 eye = make_float3(0, 0, 4);
    float testX = 2 * tid.x;
    float testY = 2 * tid.y;
    testX -= deviceConstant.viewPort.x;
    testY -= deviceConstant.viewPort.y;
    uint bla = min(deviceConstant.viewPort.x, deviceConstant.viewPort.y);
    testX /= bla;
    testY /= bla;
    testX *= deviceConstant.zoom;
    testY *= deviceConstant.zoom;
    float3 rayDir = make_float3(testX, testY, 0) - eye;

    eye = mul(deviceConstant.rotation, eye);
    rayDir = mul(deviceConstant.rotation, rayDir);

    float3 t1 = fmaxf(((-1 - eye) / rayDir), make_float3(0, 0, 0));
    float3 t2 = fmaxf(((1 - eye) / rayDir), make_float3(0, 0, 0));

    float3 front = fminf(t1, t2);
    float3 back = fmaxf(t1, t2);

    float tfront = fmaxf(front.x, fmaxf(front.y, front.z));
    float tback = fminf(back.x, fminf(back.y, back.z));

    float3 texf = (eye + tfront * rayDir + 1) / 2;
    float3 texb = (eye + tback * rayDir + 1) / 2;

    float steps = floor(length(texf - texb) * deviceConstant.volumeSize.x + 0.5f);

    float3 texDir = (texb - texf) / steps;

    steps = (tfront >= tback) ? 0 : steps;

    float m = 0;
    for (float i = 0.5f; i < steps; i++) {
        float3 sam = texf + i * texDir;
        float s = tex3D<float>(speedSizeInTex, sam.x, sam.y, sam.z);
        m = fmaxf(m, s);
    }

    float4 color = (steps > 100) ? (lerp(make_float4(0.0f, -1.41f, -3.0f, -0.4f), make_float4(1.41f, 1.41f, 1.0f, 1.41f), m / 3.0f))
                                 : (lerp(make_float4(0.0f, -1.41f, -3.0f, -0.4f), make_float4(0.0f, 1.41f, 1.0f, 1.41f), m / 3.0f));

#if USE_TEXTURE_2D
    surf2Dwrite(color, viewOutput, tid.x * sizeof(float4), tid.y);
#else
    viewOutput[tid.y * deviceConstant.viewPort.x + tid.x] = rgbaFloatToInt(color);
#endif
}

} // namespace Kernel

#endif