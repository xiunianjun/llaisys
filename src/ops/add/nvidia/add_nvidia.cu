#include "add_nvidia.hpp"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace {
void checkCudaError(cudaError_t err, const char *api) {
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] " << api << " failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(api);
    }
}

template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        c[idx] = a[idx] + b[idx];
    }
}

template <>
__global__ void addKernel<__half>(__half *c, const __half *a, const __half *b, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

template <>
__global__ void addKernel<__nv_bfloat16>(__nv_bfloat16 *c, const __nv_bfloat16 *a, const __nv_bfloat16 *b,
                                         size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        c[idx] = __hadd(a[idx], b[idx]);
    }
}

template <typename T>
void launchAdd(std::byte *c, const std::byte *a, const std::byte *b, size_t numel) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    addKernel<<<grid_size, block_size>>>(reinterpret_cast<T *>(c), reinterpret_cast<const T *>(a),
                                         reinterpret_cast<const T *>(b), numel);
    checkCudaError(cudaGetLastError(), "addKernel");
}
} // namespace

namespace llaisys::ops::nvidia {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchAdd<float>(c, a, b, numel);
    case LLAISYS_DTYPE_F16:
        return launchAdd<__half>(c, a, b, numel);
    case LLAISYS_DTYPE_BF16:
        return launchAdd<__nv_bfloat16>(c, a, b, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
