#pragma once

#include "../../tensor/tensor.hpp"

// vals:    [numel]
// max_idx: [1], int64 index of maximum value
// max_val: [1], maximum value
namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
}
