#include "rope_nvidia.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, size_t seq_len,
          size_t n_heads, size_t head_dim, float theta) {
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops::nvidia
