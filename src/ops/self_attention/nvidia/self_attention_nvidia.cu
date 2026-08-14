#include "self_attention_nvidia.hpp"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void checkCudaError(cudaError_t err, const char *api) {
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] " << api << " failed: " << cudaGetErrorString(err) << std::endl;
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
__global__ void selfAttentionKernel(T *attn_val, const T *q, const T *k, const T *v, size_t q_len, size_t kv_len,
                                    size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t value_dim,
                                    float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numel = q_len * n_heads * value_dim;
    if (idx >= numel) {
        return;
    }

    size_t y = idx % value_dim;
    size_t h = (idx / value_dim) % n_heads;
    size_t i = idx / (value_dim * n_heads);
    size_t kv_h = h / (n_heads / n_kv_heads);
    size_t causal_limit = i + kv_len - q_len;

    float max_score = -INFINITY;
    for (size_t j = 0; j < kv_len; j++) {
        if (j > causal_limit) {
            continue;
        }

        float score = 0.0f;
        for (size_t x = 0; x < head_dim; x++) {
            score += toFloat(q[(i * n_heads + h) * head_dim + x]) *
                     toFloat(k[(j * n_kv_heads + kv_h) * head_dim + x]);
        }
        score *= scale;
        max_score = fmaxf(max_score, score);
    }

    float denominator = 0.0f;
    float result = 0.0f;
    for (size_t j = 0; j < kv_len; j++) {
        if (j > causal_limit) {
            continue;
        }

        float score = 0.0f;
        for (size_t x = 0; x < head_dim; x++) {
            score += toFloat(q[(i * n_heads + h) * head_dim + x]) *
                     toFloat(k[(j * n_kv_heads + kv_h) * head_dim + x]);
        }
        float exp_score = expf(score * scale - max_score);
        denominator += exp_score;
        result += exp_score * toFloat(v[(j * n_kv_heads + kv_h) * value_dim + y]);
    }

    attn_val[(i * n_heads + h) * value_dim + y] = fromFloat<T>(result / denominator);
}

template <typename T>
void launchSelfAttention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, size_t q_len,
                         size_t kv_len, size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t value_dim,
                         float scale) {
    constexpr int block_size = 256;
    size_t numel = q_len * n_heads * value_dim;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    selfAttentionKernel<<<grid_size, block_size>>>(reinterpret_cast<T *>(attn_val), reinterpret_cast<const T *>(q),
                                                   reinterpret_cast<const T *>(k), reinterpret_cast<const T *>(v),
                                                   q_len, kv_len, n_heads, n_kv_heads, head_dim, value_dim, scale);
    checkCudaError(cudaGetLastError(), "selfAttentionKernel");
}
} // namespace

namespace llaisys::ops::nvidia {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t q_len, size_t kv_len, size_t n_heads, size_t n_kv_heads,
                    size_t head_dim, size_t value_dim, float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchSelfAttention<float>(attn_val, q, k, v, q_len, kv_len, n_heads, n_kv_heads, head_dim, value_dim,
                                          scale);
    case LLAISYS_DTYPE_F16:
        return launchSelfAttention<__half>(attn_val, q, k, v, q_len, kv_len, n_heads, n_kv_heads, head_dim, value_dim,
                                           scale);
    case LLAISYS_DTYPE_BF16:
        return launchSelfAttention<__nv_bfloat16>(attn_val, q, k, v, q_len, kv_len, n_heads, n_kv_heads, head_dim,
                                                  value_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
