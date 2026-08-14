#include "argmax_nvidia.hpp"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
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
__device__ bool better(T candidate_val, int64_t candidate_idx, T current_val, int64_t current_idx) {
    if (current_idx < 0) {
        return true;
    }

    float candidate = toFloat(candidate_val);
    float current = toFloat(current_val);
    return candidate > current || (candidate == current && candidate_idx < current_idx);
}

template <typename T>
__global__ void setEmptyArgmaxKernel(int64_t *max_idx, T *max_val) {
    *max_idx = -1;
    *max_val = T(0.0f);
}

template <typename T>
__global__ void argmaxReduceKernel(int64_t *out_idxs, T *out_vals, const int64_t *in_idxs, const T *in_vals,
                                   size_t numel) {
    extern __shared__ unsigned char shared[];
    T *shared_vals = reinterpret_cast<T *>(shared);
    size_t vals_bytes = blockDim.x * sizeof(T);
    size_t idx_offset = (vals_bytes + alignof(int64_t) - 1) & ~(alignof(int64_t) - 1);
    int64_t *shared_idxs = reinterpret_cast<int64_t *>(shared + idx_offset);

    size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    T best_val = T(0.0f);
    int64_t best_idx = -1;

    // 和相邻的比较
    if (i < numel) {
        best_val = in_vals[i];
        best_idx = in_idxs ? in_idxs[i] : static_cast<int64_t>(i);
    }

    if (i + 1 < numel) {
        T candidate_val = in_vals[i + 1];
        int64_t candidate_idx = in_idxs ? in_idxs[i + 1] : static_cast<int64_t>(i + 1);
        if (better(candidate_val, candidate_idx, best_val, best_idx)) {
            best_val = candidate_val;
            best_idx = candidate_idx;
        }
    }

    shared_vals[threadIdx.x] = best_val;
    shared_idxs[threadIdx.x] = best_idx;
    __syncthreads();

    // 每个 thread 处理 idx 和 idx + stride 的比较，一直 reduce 直到只剩一个元素
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride &&
            better(shared_vals[threadIdx.x + stride], shared_idxs[threadIdx.x + stride], shared_vals[threadIdx.x],
                   shared_idxs[threadIdx.x])) {
            shared_vals[threadIdx.x] = shared_vals[threadIdx.x + stride];
            shared_idxs[threadIdx.x] = shared_idxs[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // 当前 block 最大数值计算完毕，由 thread 0 写出
    if (threadIdx.x == 0) {
        out_vals[blockIdx.x] = shared_vals[0];
        out_idxs[blockIdx.x] = shared_idxs[0];
    }
}

template <typename T>
void launchArgmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t vals_size) {
    constexpr int block_size = 256;
    size_t shared_size = block_size * sizeof(T) + block_size * sizeof(int64_t) + alignof(int64_t);
    auto *out_idx = reinterpret_cast<int64_t *>(max_idx);
    auto *out_val = reinterpret_cast<T *>(max_val);
    auto *in_vals = reinterpret_cast<const T *>(vals);

    if (vals_size == 0) {
        setEmptyArgmaxKernel<<<1, 1>>>(out_idx, out_val);
        checkCudaError(cudaGetLastError(), "setEmptyArgmaxKernel");
        return;
    }

    size_t block_count = (vals_size + block_size * 2 - 1) / (block_size * 2);
    if (block_count == 1) {
        argmaxReduceKernel<<<1, block_size, shared_size>>>(out_idx, out_val, nullptr, in_vals, vals_size);
        checkCudaError(cudaGetLastError(), "argmaxReduceKernel");
        return;
    }

    // 保存每个 block 内部的最大值及其下标
    int64_t *current_idxs = nullptr;
    T *current_vals = nullptr;
    checkCudaError(cudaMalloc(&current_idxs, block_count * sizeof(int64_t)), "cudaMalloc");
    checkCudaError(cudaMalloc(&current_vals, block_count * sizeof(T)), "cudaMalloc");

    argmaxReduceKernel<<<static_cast<int>(block_count), block_size, shared_size>>>(current_idxs, current_vals, nullptr,
                                                                                  in_vals, vals_size);
    checkCudaError(cudaGetLastError(), "argmaxReduceKernel");

    // __syncthreads 只能处理 block 内部的 sync，所以对于涵盖多个 block 的场景，需要对 grid size 维度再做一次 reduce
    size_t current_count = block_count;
    while (current_count > 1) {
        size_t next_count = (current_count + block_size * 2 - 1) / (block_size * 2);

        if (next_count == 1) {
            argmaxReduceKernel<<<1, block_size, shared_size>>>(out_idx, out_val, current_idxs, current_vals,
                                                               current_count);
            checkCudaError(cudaGetLastError(), "argmaxReduceKernel");
            checkCudaError(cudaFree(current_idxs), "cudaFree");
            checkCudaError(cudaFree(current_vals), "cudaFree");
            return;
        }

        int64_t *next_idxs = nullptr;
        T *next_vals = nullptr;
        checkCudaError(cudaMalloc(&next_idxs, next_count * sizeof(int64_t)), "cudaMalloc");
        checkCudaError(cudaMalloc(&next_vals, next_count * sizeof(T)), "cudaMalloc");

        argmaxReduceKernel<<<static_cast<int>(next_count), block_size, shared_size>>>(next_idxs, next_vals,
                                                                                     current_idxs, current_vals,
                                                                                     current_count);
        checkCudaError(cudaGetLastError(), "argmaxReduceKernel");

        checkCudaError(cudaFree(current_idxs), "cudaFree");
        checkCudaError(cudaFree(current_vals), "cudaFree");
        current_idxs = next_idxs;
        current_vals = next_vals;
        current_count = next_count;
    }
}
} // namespace

namespace llaisys::ops::nvidia {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t vals_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchArgmax<float>(max_idx, max_val, vals, vals_size);
    case LLAISYS_DTYPE_F16:
        return launchArgmax<__half>(max_idx, max_val, vals, vals_size);
    case LLAISYS_DTYPE_BF16:
        return launchArgmax<__nv_bfloat16>(max_idx, max_val, vals, vals_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
