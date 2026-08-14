#include "swiglu_nvidia.hpp"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {
template <typename T>
void launchSwiGLU(std::byte *out, const std::byte *gate, const std::byte *up, size_t numel) {
    TO_BE_IMPLEMENTED();
}
} // namespace

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchSwiGLU<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return launchSwiGLU<__half>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return launchSwiGLU<__nv_bfloat16>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
