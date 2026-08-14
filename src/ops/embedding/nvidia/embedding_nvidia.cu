#include "embedding_nvidia.hpp"

#include "../../../utils.hpp"

namespace {
void launchEmbedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t index_size,
                     size_t num_embeddings, size_t row_bytes) {
    TO_BE_IMPLEMENTED();
}
} // namespace

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t index_size,
               size_t num_embeddings, size_t row_bytes) {
    return launchEmbedding(out, index, weight, index_size, num_embeddings, row_bytes);
}
} // namespace llaisys::ops::nvidia
