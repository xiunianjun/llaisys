#include "rms_norm_nvidia.hpp"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {
template <typename T>
void launchRmsNorm(std::byte *out, const std::byte *in, const std::byte *weight, size_t rows, size_t cols, float eps) {
    TO_BE_IMPLEMENTED();
}
} // namespace

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, size_t rows,
              size_t cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchRmsNorm<float>(out, in, weight, rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return launchRmsNorm<__half>(out, in, weight, rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return launchRmsNorm<__nv_bfloat16>(out, in, weight, rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
