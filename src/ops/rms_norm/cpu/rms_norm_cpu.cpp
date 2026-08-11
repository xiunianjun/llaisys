#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t cols, float eps) {
    for (size_t r = 0; r < rows; r++) {
        float denominator = 0;
        for (size_t c = 0; c < cols; c++) {
            float x = llaisys::utils::cast<float>(in[r * cols + c]);
            denominator += x * x;
        }
        denominator = std::sqrt(denominator / cols + eps);

        for (size_t c = 0; c < cols; c++) {
            out[r * cols + c] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(weight[c]) * llaisys::utils::cast<float>(in[r * cols + c]) / denominator);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type, size_t rows,
              size_t cols, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), rows, cols, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), rows, cols, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), rows, cols, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
