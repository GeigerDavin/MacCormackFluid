#ifndef KERNEL_TEXTURES_HPP
#define KERNEL_TEXTURES_HPP

namespace CUDA {

texture<float4, cudaTextureType3D> speed;
texture<float, cudaTextureType3D> speedSize;

texture<float4, cudaTextureType3D> divergence;
texture<float4, cudaTextureType3D> pressure;

template <class Texture>
inline void bindTexture(const Texture& tex, const void* ptr, size_t size) {
	if (size > 0 && useCuda) {
		cudaBindTexture(nullptr, tex, ptr, size);
		ERRORCHECK_CUDA();
	}
}

template <class Texture>
void unbindTexture(const Texture& tex) {
	if (useCuda) {
		cudaUnbindTexture(tex);
		ERRORCHECK_CUDA();
	}
}

} // namespace CUDA

#endif
