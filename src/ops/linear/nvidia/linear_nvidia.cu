#include "linear_nvidia.hpp"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {
template <typename T>
void launchLinear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t batch,
                  size_t out_features, size_t in_features) {
    TO_BE_IMPLEMENTED();
}
} // namespace

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch, size_t out_features, size_t in_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchLinear<float>(out, in, weight, bias, batch, out_features, in_features);
    case LLAISYS_DTYPE_F16:
        return launchLinear<__half>(out, in, weight, bias, batch, out_features, in_features);
    case LLAISYS_DTYPE_BF16:
        return launchLinear<__nv_bfloat16>(out, in, weight, bias, batch, out_features, in_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
