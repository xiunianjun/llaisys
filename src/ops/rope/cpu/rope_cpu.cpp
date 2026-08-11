#include "rope_cpu.hpp"

#include "../../../utils.hpp"
#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, size_t seq_len, size_t n_heads, size_t head_dim, float theta) {
    for (size_t seq = 0; seq < seq_len; seq++) {
        for (size_t head = 0; head < n_heads; head++) {
            for (size_t dim = 0; dim < head_dim / 2; dim++) {
                float fi = static_cast<float>(pos_ids[seq]) / std::pow(theta, 2.0f * dim / head_dim);
                float sin_fi = std::sin(fi);
                float cos_fi = std::cos(fi);
                float a = llaisys::utils::cast<float>(in[dim]);
                float b = llaisys::utils::cast<float>(in[dim + head_dim / 2]);
                out[dim] = llaisys::utils::cast<T>(a * cos_fi - b * sin_fi);
                out[dim + head_dim / 2] = llaisys::utils::cast<T>(b * cos_fi + a * sin_fi);
            }
            out += head_dim;
            in += head_dim;
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, size_t seq_len,
          size_t n_heads, size_t head_dim, float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seq_len, n_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), seq_len, n_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
