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
    //r.w = dot(v, m.m[3]);
    r.w = 0.0f;
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
    TID_CONST;
    float3 eye = make_float3(0, 0, 4);
    float rayX = 2 * g.mouse.x;
    float rayY = 2 * g.mouse.y;
//    rayX -= (float) 1280;
//    rayY -= (float) 720;
//    float viewPort = fminf(1280, 720);

    rayX -= (float) g.viewPort.x;
    rayY -= (float) g.viewPort.y;
    float viewPort = fminf(g.viewPort.x, g.viewPort.y);
    rayX /= viewPort;
    rayY /= viewPort;
    rayX *= g.zoom;
    rayY *= g.zoom;
    float3 rayDir = make_float3(rayX, rayY, 0) - eye;

    if (tid.x == 0 && tid.y == 0 && tid.z == 0) {
    	//printf("%u\n", g.viewPort.x);
    }

    float3 viewDir = 0 - eye;

    float t = -dot(viewDir, eye) / dot(viewDir, rayDir);

    float3 ball = eye + t * rayDir;

    ball = mul(g.rotation, ball);
    float4 force = mul(g.rotation, g.dragDirection) * g.zoom * 4;

    // Texture coordinates of intersection
    ball.x = volumeSizeDev.x * (ball.x + 1) / 2;
    ball.y = volumeSizeDev.y * (ball.y + 1) / 2;
    ball.z = volumeSizeDev.z * (ball.z + 1) / 2;

    // Volume coordinates relative to sphere center
    ball.x = tid.x - ball.x;
    ball.y = tid.y - ball.y;
    ball.z = tid.z - ball.z;

    float r = dot(ball, ball); // Radius^2 square

    // Draw sphere
    float scale = exp(-r / (g.mouse.w * g.mouse.w));

    float4 speed = surf3Dread<float4>(speedIn, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);

    float4 sphere = make_float4(speed.x + force.x * scale,
                                speed.y + force.y * scale,
                                speed.z + force.z * scale,
                                speed.w + length(force) * scale);

//    if (sphere.w > 0) {
//    	printf("%f\n", sphere.w);
//    }
//
//    if (sphere.w > 0.1f) {
//    	printf("%.16f \r", sphere.w);
//    }

    surf3Dwrite(sphere, speedOut, tid.x * sizeof(float4), tid.y, tid.z, CUDA_BOUNDARY_MODE);
}

__global__ static void renderVolume(cudaTextureObject_t speedSizeInTex,
                                    uint* viewOutput)
{
    TID_CONST;
    float3 eye = make_float3(0, 0, 4);
    float rayX = 2 * tid.x;
    float rayY = 2 * tid.y;
    rayX -= g.viewPort.x;
    rayY -= g.viewPort.y;
    uint viewPort = min(g.viewPort.x, g.viewPort.y);
    rayX /= viewPort;
    rayY /= viewPort;
    rayX *= g.zoom;
    rayY *= g.zoom;
    float3 rayDir = make_float3(rayX, rayY, 0) - eye;

    eye = mul(g.rotation, eye);
    rayDir = mul(g.rotation, rayDir);

    float3 t1 = fmaxf(((-1 - eye) / rayDir), make_float3(0, 0, 0));
    float3 t2 = fmaxf(((1 - eye) / rayDir), make_float3(0, 0, 0));

    float3 front = fminf(t1, t2);
    float3 back = fmaxf(t1, t2);

    float tfront = fmaxf(front.x, fmaxf(front.y, front.z));
    float tback = fminf(back.x, fminf(back.y, back.z));

    float3 texf = (eye + tfront * rayDir + 1) / 2;
    float3 texb = (eye + tback * rayDir + 1) / 2;

    float steps = floor(length(texf - texb) * volumeSizeDev.x + 0.5f);

    float3 texDir = (texb - texf) / steps;

    steps = (tfront >= tback) ? 0 : steps;

    float m = 0;
    for (float i = 0.5f; i < steps; i++) {
        float3 sam = texf + i * texDir;
        float s = tex3D<float>(speedSizeInTex, sam.x, sam.y, sam.z);
        m = fmaxf(m, s);
    }

    float4 color = (lerp(make_float4(0.0f, -1.41f, -3.0f, -0.4f), make_float4(1.41f, 1.41f+workerId, 1.0f, 1.41f), m / 3.0f));

    viewOutput[tid.y * g.viewPort.x + tid.x] = rgbaFloatToInt(color);
}

} // namespace Kernel

#endif
