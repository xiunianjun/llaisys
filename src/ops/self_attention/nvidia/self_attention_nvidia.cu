#include "self_attention_nvidia.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, size_t q_len, size_t kv_len, size_t n_heads, size_t n_kv_heads,
                    size_t head_dim, size_t value_dim, float scale) {
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops::nvidia
