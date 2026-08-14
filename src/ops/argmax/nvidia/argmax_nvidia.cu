#include "argmax_nvidia.hpp"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {
template <typename T>
void launchArgmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t vals_size) {
    TO_BE_IMPLEMENTED();
}
} // namespace

namespace llaisys::ops::nvidia {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t vals_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchArgmax<float>(max_idx, max_val, vals, vals_size);
    case LLAISYS_DTYPE_F16:
        return launchArgmax<__half>(max_idx, max_val, vals, vals_size);
    case LLAISYS_DTYPE_BF16:
        return launchArgmax<__nv_bfloat16>(max_idx, max_val, vals, vals_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
