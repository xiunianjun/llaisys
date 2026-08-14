#include "embedding_nvidia.hpp"

#include "../../../utils.hpp"

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

__global__ void embeddingKernel(std::byte *out, const int64_t *index, const std::byte *weight, size_t index_size,
                                size_t num_embeddings, size_t row_bytes) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_bytes = index_size * row_bytes;
    if (idx < total_bytes) {
        size_t row = idx / row_bytes;
        size_t col_byte = idx % row_bytes;
        int64_t src_row = index[row];

        if (src_row >= 0 && static_cast<size_t>(src_row) < num_embeddings) {
            out[idx] = weight[static_cast<size_t>(src_row) * row_bytes + col_byte];
        }
    }
}

void launchEmbedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t index_size,
                     size_t num_embeddings, size_t row_bytes) {
    constexpr int block_size = 256;
    size_t total_bytes = index_size * row_bytes;
    int grid_size = static_cast<int>((total_bytes + block_size - 1) / block_size);

    embeddingKernel<<<grid_size, block_size>>>(out, reinterpret_cast<const int64_t *>(index), weight, index_size,
                                               num_embeddings, row_bytes);
    checkCudaError(cudaGetLastError(), "embeddingKernel");
}
} // namespace

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t index_size,
               size_t num_embeddings, size_t row_bytes) {
    return launchEmbedding(out, index, weight, index_size, num_embeddings, row_bytes);
}
} // namespace llaisys::ops::nvidia
