#pragma once

#include "../../tensor/tensor.hpp"

// out:     [seq_len, n_heads, head_dim]
// in:      [seq_len, n_heads, head_dim]
// pos_ids: [seq_len], int64 positions
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
}
