#include "embedding_nvidia.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t index_size,
               size_t num_embeddings, size_t row_bytes) {
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops::nvidia
