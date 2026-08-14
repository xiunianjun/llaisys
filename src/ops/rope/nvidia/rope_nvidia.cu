#include "rope_nvidia.hpp"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace {
template <typename T>
void launchRope(std::byte *out, const std::byte *in, const std::byte *pos_ids, size_t seq_len, size_t n_heads,
                size_t head_dim, float theta) {
    TO_BE_IMPLEMENTED();
}
} // namespace

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, size_t seq_len,
          size_t n_heads, size_t head_dim, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchRope<float>(out, in, pos_ids, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return launchRope<__half>(out, in, pos_ids, seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return launchRope<__nv_bfloat16>(out, in, pos_ids, seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
