#pragma once

#include "../../tensor/tensor.hpp"

// out:    [batch, out_features]
// in:     [batch, in_features]
// weight: [out_features, in_features]
// bias:   [out_features], optional
namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
}
