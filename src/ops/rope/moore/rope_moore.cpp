#include "rope_moore.hpp"

#include "../../../utils.hpp"

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <cstdint>
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
__global__ void ropeKernel(T *out, const T *in, const int64_t *pos_ids, size_t seq_len, size_t n_heads,
                           size_t head_dim, float theta) {
    size_t half_dim = head_dim / 2;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numel = seq_len * n_heads * half_dim;
    if (idx >= numel) {
        return;
    }

    size_t dim = idx % half_dim;
    size_t head = (idx / half_dim) % n_heads;
    size_t seq = idx / (half_dim * n_heads);
    size_t base = (seq * n_heads + head) * head_dim;

    float fi = static_cast<float>(pos_ids[seq]) / powf(theta, 2.0f * static_cast<float>(dim) / head_dim);
    float sin_fi = sinf(fi);
    float cos_fi = cosf(fi);
    float a = toFloat(in[base + dim]);
    float b = toFloat(in[base + dim + half_dim]);

    out[base + dim] = fromFloat<T>(a * cos_fi - b * sin_fi);
    out[base + dim + half_dim] = fromFloat<T>(b * cos_fi + a * sin_fi);
}

template <typename T>
void launchRope(std::byte *out, const std::byte *in, const std::byte *pos_ids, size_t seq_len, size_t n_heads,
                size_t head_dim, float theta) {
    constexpr int block_size = 256;
    size_t work_items = seq_len * n_heads * (head_dim / 2);
    int grid_size = static_cast<int>((work_items + block_size - 1) / block_size);

    ropeKernel<<<grid_size, block_size>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(in),
                                          reinterpret_cast<const int64_t *>(pos_ids), seq_len, n_heads, head_dim,
                                          theta);
    checkCudaError(musaGetLastError(), "ropeKernel");
}
} // namespace

namespace llaisys::ops::moore {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, size_t seq_len,
          size_t n_heads, size_t head_dim, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchRope<float>(out, in, pos_ids, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return launchRope<__half>(out, in, pos_ids, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return launchRope<__nv_bfloat16>(out, in, pos_ids, seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::moore
