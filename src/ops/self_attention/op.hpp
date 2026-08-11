#pragma once

#include "../../tensor/tensor.hpp"

// attn_val: [seq_len,   n_heads,    value_dim]
// q:        [seq_len,   n_heads,    head_dim]
// k:        [total_len, n_kv_heads, head_dim]
// v:        [total_len, n_kv_heads, value_dim]
namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
}
