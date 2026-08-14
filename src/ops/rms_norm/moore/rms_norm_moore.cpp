#include "rms_norm_moore.hpp"

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
__global__ void rmsNormKernel(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    extern __shared__ float shared[];
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;

    float sum = 0.0f;
    for (size_t col = tid; col < cols; col += blockDim.x) {
        float x = toFloat(in[row * cols + col]);
        sum += x * x;
    }

    shared[tid] = sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float inv_rms = rsqrtf(shared[0] / static_cast<float>(cols) + eps);
    for (size_t col = tid; col < cols; col += blockDim.x) {
        float y = toFloat(in[row * cols + col]) * inv_rms * toFloat(weight[col]);
        out[row * cols + col] = fromFloat<T>(y);
    }
}

template <typename T>
void launchRmsNorm(std::byte *out, const std::byte *in, const std::byte *weight, size_t rows, size_t cols, float eps) {
    constexpr int block_size = 256;
    rmsNormKernel<<<static_cast<int>(rows), block_size, block_size * sizeof(float)>>>(
        reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), reinterpret_cast<const T *>(weight), rows, cols,
        eps);
    checkCudaError(musaGetLastError(), "rmsNormKernel");
}
} // namespace

namespace llaisys::ops::moore {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, size_t rows,
              size_t cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchRmsNorm<float>(out, in, weight, rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return launchRmsNorm<__half>(out, in, weight, rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return launchRmsNorm<__nv_bfloat16>(out, in, weight, rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::moore
