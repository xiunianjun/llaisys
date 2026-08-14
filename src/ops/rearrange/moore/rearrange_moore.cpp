#include "rearrange_moore.hpp"

#include "../../../utils.hpp"

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <iostream>
#include <stdexcept>

namespace {
void checkCudaError(musaError_t err, const char *api) {
    if (err != musaSuccess) {
        std::cerr << "[ERROR] " << api << " failed: " << musaGetErrorString(err) << std::endl;
        throw std::runtime_error(api);
    }
}

template <typename T>
__global__ void rearrangeKernel(T *out, const T *in, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = in[idx];
    }
}

template <typename T>
void launchRearrange(std::byte *out, const std::byte *in, size_t numel) {
    constexpr int block_size = 256;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    rearrangeKernel<<<grid_size, block_size>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in), numel);
    checkCudaError(musaGetLastError(), "rearrangeKernel");
}
} // namespace

namespace llaisys::ops::moore {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchRearrange<float>(out, in, numel);
    case LLAISYS_DTYPE_F16:
        return launchRearrange<__half>(out, in, numel);
    case LLAISYS_DTYPE_BF16:
        return launchRearrange<__nv_bfloat16>(out, in, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::moore
