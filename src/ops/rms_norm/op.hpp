#pragma once

#include "../../tensor/tensor.hpp"

// out:    [rows, hidden_size]
// in:     [rows, hidden_size]
// weight: [hidden_size]
namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
}
