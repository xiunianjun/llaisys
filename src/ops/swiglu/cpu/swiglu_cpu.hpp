#pragma once
#include "llaisys.h"

#include <cstddef>

// gate.shape = [seqlen, intermediate_size]
// up.shape   = [seqlen, intermediate_size]
// out.shape  = [seqlen, intermediate_size]
namespace llaisys::ops::cpu {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel);
}
