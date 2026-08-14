#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::moore {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, size_t numel);
}
