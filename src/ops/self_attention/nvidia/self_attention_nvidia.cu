#include "self_attention_nvidia.hpp"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {
template <typename T>
void launchSelfAttention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, size_t q_len,
                         size_t kv_len, size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t value_dim,
                         float scale) {
    TO_BE_IMPLEMENTED();
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
