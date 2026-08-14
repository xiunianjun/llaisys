#include "rearrange_nvidia.hpp"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {
template <typename T>
void launchRearrange(std::byte *out, const std::byte *in, size_t numel) {
    TO_BE_IMPLEMENTED();
}
} // namespace

namespace llaisys::ops::nvidia {
void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchRearrange<float>(out, in, numel);
    case LLAISYS_DTYPE_F16:
        return launchRearrange<__half>(out, in, numel);
    case LLAISYS_DTYPE_BF16:
        return launchRearrange<__nv_bfloat16>(out, in, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
