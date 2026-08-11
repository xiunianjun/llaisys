#pragma once

#include "../../tensor/tensor.hpp"

// out:  [seq_len, intermediate_size]
// gate: [seq_len, intermediate_size]
// up:   [seq_len, intermediate_size]
namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up);
}
