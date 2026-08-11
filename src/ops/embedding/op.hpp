#pragma once

#include "../../tensor/tensor.hpp"

// out:    [num_tokens, hidden_size]
// index:  [num_tokens], int64 row ids
// weight: [num_embeddings, hidden_size]
namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight);
}
