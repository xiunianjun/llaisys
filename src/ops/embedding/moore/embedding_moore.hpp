#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::moore {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, size_t index_size,
               size_t num_embeddings, size_t row_bytes);
}
