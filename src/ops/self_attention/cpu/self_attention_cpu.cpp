#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, size_t q_len, size_t kv_len, size_t n_heads,
                     size_t n_kv_heads, size_t head_dim, size_t value_dim, float scale) {
    std::vector<float> A(n_heads * q_len * kv_len);

    // calculate A = QK^T * scale, with causal mask
    for (size_t h = 0; h < n_heads; h++) {
        size_t kv_h = h / (n_heads / n_kv_heads);
        for (size_t i = 0; i < q_len; i++) {
            for (size_t j = 0; j < kv_len; j++) {
                if (j > i + kv_len - q_len) {
                    A[(h * q_len + i) * kv_len + j] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                float score = 0.0f;
                for (size_t x = 0; x < head_dim; x++) {
                    score += llaisys::utils::cast<float>(q[(i * n_heads + h) * head_dim + x])
                             * llaisys::utils::cast<float>(k[(j * n_kv_heads + kv_h) * head_dim + x]);
                }
                A[(h * q_len + i) * kv_len + j] = score * scale;
            }
        }
    }

    // calculate softmax(A) along kv_len
    for (size_t h = 0; h < n_heads; h++) {
        for (size_t i = 0; i < q_len; i++) {
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < kv_len; j++) {
                max_score = std::max(max_score, A[(h * q_len + i) * kv_len + j]);
            }

            float denominator = 0.0f;
            for (size_t j = 0; j < kv_len; j++) {
                A[(h * q_len + i) * kv_len + j] = std::exp(A[(h * q_len + i) * kv_len + j] - max_score);
                denominator += A[(h * q_len + i) * kv_len + j];
            }

            for (size_t j = 0; j < kv_len; j++) {
                A[(h * q_len + i) * kv_len + j] /= denominator;
            }
        }
    }

    // calculate result = softmax(A) V
    for (size_t h = 0; h < n_heads; h++) {
        size_t kv_h = h / (n_heads / n_kv_heads);
        for (size_t i = 0; i < q_len; i++) {
            for (size_t y = 0; y < value_dim; y++) {
                float score = 0.0f;
                for (size_t j = 0; j < kv_len; j++) {
                    score += A[(h * q_len + i) * kv_len + j]
                             * llaisys::utils::cast<float>(v[(j * n_kv_heads + kv_h) * value_dim + y]);
                }
                attn_val[(i * n_heads + h) * value_dim + y] = llaisys::utils::cast<T>(score);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type,
                    size_t q_len, size_t kv_len, size_t n_heads, size_t n_kv_heads, size_t head_dim, size_t value_dim,
                    float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v), q_len, kv_len,
                               n_heads, n_kv_heads, head_dim, value_dim, scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                               q_len, kv_len, n_heads, n_kv_heads, head_dim, value_dim, scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                               q_len, kv_len, n_heads, n_kv_heads, head_dim, value_dim, scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
