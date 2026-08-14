#include "swiglu_moore.hpp"

#include "../../../utils.hpp"

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void checkCudaError(musaError_t err, const char *api) {
    if (err != musaSuccess) {
        std::cerr << "[ERROR] " << api << " failed: " << musaGetErrorString(err) << std::endl;
        throw std::runtime_error(api);
    }
}

__device__ float toFloat(float val) {
    return val;
}

__device__ float toFloat(__half val) {
    return __half2float(val);
}

__device__ float toFloat(__nv_bfloat16 val) {
    return __bfloat162float(val);
}

template <typename T>
__device__ T fromFloat(float val);

template <>
__device__ float fromFloat<float>(float val) {
    return val;
}

template <>
__device__ __half fromFloat<__half>(float val) {
    return __float2half(val);
}

template <>
__device__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float val) {
    return __float2bfloat16(val);
}

template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float gate_val = toFloat(gate[idx]);
        float up_val = toFloat(up[idx]);
        out[idx] = fromFloat<T>(up_val * gate_val / (1.0f + expf(-gate_val)));
    }
}

template <typename T>
void launchSwiGLU(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    swigluKernel<<<grid_size, block_size>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(gate),
                                            reinterpret_cast<const T *>(up), numel);
    checkCudaError(musaGetLastError(), "swigluKernel");
}
} // namespace

namespace llaisys::ops::moore {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchSwiGLU<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return launchSwiGLU<__half>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return launchSwiGLU<__nv_bfloat16>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::moore
