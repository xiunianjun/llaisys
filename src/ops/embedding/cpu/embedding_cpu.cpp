#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>
#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t index_size, size_t num_embeddings,
               size_t row_bytes) {
    const auto *idx = reinterpret_cast<const int64_t *>(index);

    for (size_t i = 0; i < index_size; i++) {
        CHECK_ARGUMENT(idx[i] >= 0 && static_cast<size_t>(idx[i]) < num_embeddings, "Embedding: index out of range.");
        std::memcpy(out + i * row_bytes, weight + static_cast<size_t>(idx[i]) * row_bytes, row_bytes);
    }
}
} // namespace llaisys::ops::cpu
