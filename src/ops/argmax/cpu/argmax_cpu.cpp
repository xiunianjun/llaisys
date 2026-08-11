#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t vals_size) {
    if (vals_size == 0) {
        *max_idx = -1;
        *max_val = llaisys::utils::cast<T>(0.0f);
        return;
    }

    *max_val = vals[0];
    *max_idx = 0;
    for (size_t i = 1; i < vals_size; i++) {
        if (llaisys::utils::cast<float>(vals[i]) > llaisys::utils::cast<float>(*max_val)) {
            *max_val = vals[i];
            *max_idx = static_cast<int64_t>(i);
        }
    }
}

namespace llaisys::ops::cpu {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t vals_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val), reinterpret_cast<const float *>(vals), vals_size);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                    reinterpret_cast<const llaisys::bf16_t *>(vals), vals_size);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                    reinterpret_cast<const llaisys::fp16_t *>(vals), vals_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
