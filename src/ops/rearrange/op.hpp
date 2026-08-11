#pragma once

#include "../../tensor/tensor.hpp"

// out: same numel as in
// in:  source tensor to rearrange
namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in);
}
