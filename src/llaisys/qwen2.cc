#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

#include "../core/llaisys_core.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>
#include <cmath>

struct LlaisysQwen2Model {
    LlaisysQwen2Meta meta{};
    LlaisysQwen2Weights weights{};
    llaisysTensor_t *k_cache{};
    llaisysTensor_t *v_cache{};
    size_t cached_len{};
    size_t cache_capacity{};
    llaisysDeviceType_t device{};
    std::vector<int> device_ids;
};

namespace {

llaisysTensor_t createTensor(
    const std::vector<size_t> &shape,
    llaisysDataType_t dtype,
    llaisysDeviceType_t device,
    int device_id) {
    return tensorCreate(
        const_cast<size_t *>(shape.data()),
        shape.size(),
        dtype,
        device,
        device_id);
}

void destroyTensor(llaisysTensor_t &tensor) {
    if (tensor != nullptr) {
        tensorDestroy(tensor);
        tensor = nullptr;
    }
}

void destroyTensorArray(llaisysTensor_t *&tensors, size_t count) {
    if (tensors == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        destroyTensor(tensors[i]);
    }
    delete[] tensors;
    tensors = nullptr;
}

llaisysTensor_t *createTensorArray(size_t count) {
    return new llaisysTensor_t[count]{};
}

void copyTensor(llaisys::tensor_t dst, llaisys::tensor_t src) {
    CHECK_SAME_SHAPE(dst->shape(), src->shape());
    CHECK_ARGUMENT(dst->dtype() == src->dtype(), "copyTensor: dtype mismatch");
    ASSERT(dst->isContiguous() && src->isContiguous(), "copyTensor: tensors must be contiguous.");

    llaisysMemcpyKind_t kind;
    if (src->deviceType() == LLAISYS_DEVICE_CPU && dst->deviceType() == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2H;
        llaisys::core::context().setDevice(LLAISYS_DEVICE_CPU, 0);
    } else if (src->deviceType() == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_H2D;
        llaisys::core::context().setDevice(dst->deviceType(), dst->deviceId());
    } else if (dst->deviceType() == LLAISYS_DEVICE_CPU) {
        kind = LLAISYS_MEMCPY_D2H;
        llaisys::core::context().setDevice(src->deviceType(), src->deviceId());
    } else {
        kind = LLAISYS_MEMCPY_D2D;
        llaisys::core::context().setDevice(dst->deviceType(), dst->deviceId());
    }

    llaisys::core::context().runtime().api()->memcpy_sync(
        dst->data(),
        src->data(),
        src->numel() * src->elementSize(),
        kind);
}

void createWeightTensors(LlaisysQwen2Model *model) {
    auto &meta = model->meta;
    auto &weights = model->weights;
    int device_id = model->device_ids.empty() ? 0 : model->device_ids[0];

    weights.in_embed = createTensor({meta.voc, meta.hs}, meta.dtype, model->device, device_id);
    weights.out_embed = createTensor({meta.voc, meta.hs}, meta.dtype, model->device, device_id);
    weights.out_norm_w = createTensor({meta.hs}, meta.dtype, model->device, device_id);

    weights.attn_norm_w = createTensorArray(meta.nlayer);
    weights.attn_q_w = createTensorArray(meta.nlayer);
    weights.attn_q_b = createTensorArray(meta.nlayer);
    weights.attn_k_w = createTensorArray(meta.nlayer);
    weights.attn_k_b = createTensorArray(meta.nlayer);
    weights.attn_v_w = createTensorArray(meta.nlayer);
    weights.attn_v_b = createTensorArray(meta.nlayer);
    weights.attn_o_w = createTensorArray(meta.nlayer);
    weights.mlp_norm_w = createTensorArray(meta.nlayer);
    weights.mlp_gate_w = createTensorArray(meta.nlayer);
    weights.mlp_up_w = createTensorArray(meta.nlayer);
    weights.mlp_down_w = createTensorArray(meta.nlayer);

    size_t q_dim = meta.nh * meta.dh;
    size_t kv_dim = meta.nkvh * meta.dh;
    for (size_t i = 0; i < meta.nlayer; ++i) {
        weights.attn_norm_w[i] = createTensor({meta.hs}, meta.dtype, model->device, device_id);
        weights.attn_q_w[i] = createTensor({q_dim, meta.hs}, meta.dtype, model->device, device_id);
        weights.attn_q_b[i] = createTensor({q_dim}, meta.dtype, model->device, device_id);
        weights.attn_k_w[i] = createTensor({kv_dim, meta.hs}, meta.dtype, model->device, device_id);
        weights.attn_k_b[i] = createTensor({kv_dim}, meta.dtype, model->device, device_id);
        weights.attn_v_w[i] = createTensor({kv_dim, meta.hs}, meta.dtype, model->device, device_id);
        weights.attn_v_b[i] = createTensor({kv_dim}, meta.dtype, model->device, device_id);
        weights.attn_o_w[i] = createTensor({meta.hs, q_dim}, meta.dtype, model->device, device_id);
        weights.mlp_norm_w[i] = createTensor({meta.hs}, meta.dtype, model->device, device_id);
        weights.mlp_gate_w[i] = createTensor({meta.di, meta.hs}, meta.dtype, model->device, device_id);
        weights.mlp_up_w[i] = createTensor({meta.di, meta.hs}, meta.dtype, model->device, device_id);
        weights.mlp_down_w[i] = createTensor({meta.hs, meta.di}, meta.dtype, model->device, device_id);
    }
}

void createCacheTensors(LlaisysQwen2Model *model) {
    auto &meta = model->meta;
    int device_id = model->device_ids.empty() ? 0 : model->device_ids[0];

    model->k_cache = createTensorArray(meta.nlayer);
    model->v_cache = createTensorArray(meta.nlayer);
    for (size_t i = 0; i < meta.nlayer; ++i) {
        model->k_cache[i] = createTensor({model->cache_capacity, meta.nkvh, meta.dh}, meta.dtype, model->device, device_id);
        model->v_cache[i] = createTensor({model->cache_capacity, meta.nkvh, meta.dh}, meta.dtype, model->device, device_id);
    }
}

void destroyWeights(LlaisysQwen2Weights &weights, size_t nlayer) {
    destroyTensor(weights.in_embed);
    destroyTensor(weights.out_embed);
    destroyTensor(weights.out_norm_w);

    destroyTensorArray(weights.attn_norm_w, nlayer);
    destroyTensorArray(weights.attn_q_w, nlayer);
    destroyTensorArray(weights.attn_q_b, nlayer);
    destroyTensorArray(weights.attn_k_w, nlayer);
    destroyTensorArray(weights.attn_k_b, nlayer);
    destroyTensorArray(weights.attn_v_w, nlayer);
    destroyTensorArray(weights.attn_v_b, nlayer);
    destroyTensorArray(weights.attn_o_w, nlayer);
    destroyTensorArray(weights.mlp_norm_w, nlayer);
    destroyTensorArray(weights.mlp_gate_w, nlayer);
    destroyTensorArray(weights.mlp_up_w, nlayer);
    destroyTensorArray(weights.mlp_down_w, nlayer);
}

void destroyCache(LlaisysQwen2Model *model) {
    destroyTensorArray(model->k_cache, model->meta.nlayer);
    destroyTensorArray(model->v_cache, model->meta.nlayer);
    model->cached_len = 0;
    model->cache_capacity = 0;
}

} // namespace

__C {

struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {
    if (meta == nullptr) {
        return nullptr;
    }

    auto *model = new LlaisysQwen2Model{};
    model->meta = *meta;
    model->device = device;
    if (device_ids != nullptr && ndevice > 0) {
        model->device_ids.assign(device_ids, device_ids + ndevice);
    } else {
        model->device_ids.push_back(0);
    }

    model->cache_capacity = std::min(model->meta.maxseq, size_t{4096});
    createWeightTensors(model);
    createCacheTensors(model);
    return model;
}

void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (model == nullptr) {
        return;
    }
    destroyCache(model);
    destroyWeights(model->weights, model->meta.nlayer);
    delete model;
}

void llaisysQwen2ModelReset(struct LlaisysQwen2Model *model) {
    if (model == nullptr) {
        return;
    }
    model->cached_len = 0;
}

struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (model == nullptr) {
        return nullptr;
    }
    return &model->weights;
}

int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken) {
    if (model == nullptr || token_ids == nullptr || ntoken == 0) {
        return int64_t{-1};
    }

    int device_id = model->device_ids.empty() ? 0 : model->device_ids[0];
    size_t q_dim = model->meta.nh * model->meta.dh;
    size_t kv_dim = model->meta.nkvh * model->meta.dh;
    size_t old_cached_len = model->cached_len;
    size_t new_cached_len = old_cached_len + ntoken;
    CHECK_ARGUMENT(new_cached_len <= model->cache_capacity, "Qwen2 KV cache capacity exceeded.");

    llaisysTensor_t input_tokens_tensor = createTensor({ntoken}, LLAISYS_DTYPE_I64, model->device, device_id);
    input_tokens_tensor->tensor->load(token_ids);

    llaisysTensor_t embedding_tensor = createTensor({ntoken, model->meta.hs}, model->meta.dtype, model->device, device_id);
    llaisysTensor_t position_ids_tensor = createTensor({ntoken}, LLAISYS_DTYPE_I64, model->device, device_id);
    std::vector<int64_t> position_ids(ntoken);
    for (size_t pos = 0; pos < ntoken; ++pos) {
        position_ids[pos] = static_cast<int64_t>(old_cached_len + pos);
    }
    position_ids_tensor->tensor->load(position_ids.data());

    // embedding
    llaisys::ops::embedding(
        embedding_tensor->tensor,
        input_tokens_tensor->tensor,
        model->weights.in_embed->tensor);

    // llaisysTensor_t *attention_vals = createTensorArray(model->meta.nlayer);

    llaisysTensor_t current_tensor = embedding_tensor;
    for (size_t i = 0; i < model->meta.nlayer; ++i) {
        llaisysTensor_t attn_residual = current_tensor;

        // RNS NORM
        llaisysTensor_t x_norm = createTensor(current_tensor->tensor->shape(), model->meta.dtype, model->device, device_id);

        llaisys::ops::rms_norm(x_norm->tensor, current_tensor->tensor, model->weights.attn_norm_w[i]->tensor, model->meta.epsilon);

        /* Attention */
        // projection
        llaisysTensor_t q_proj = createTensor({ntoken, q_dim}, model->meta.dtype, model->device, device_id);
        llaisysTensor_t k_proj = createTensor({ntoken, kv_dim}, model->meta.dtype, model->device, device_id);
        llaisysTensor_t v_proj = createTensor({ntoken, kv_dim}, model->meta.dtype, model->device, device_id);

        llaisys::ops::linear(q_proj->tensor, x_norm->tensor, model->weights.attn_q_w[i]->tensor, model->weights.attn_q_b[i]->tensor);
        llaisys::ops::linear(k_proj->tensor, x_norm->tensor, model->weights.attn_k_w[i]->tensor, model->weights.attn_k_b[i]->tensor);
        llaisys::ops::linear(v_proj->tensor, x_norm->tensor, model->weights.attn_v_w[i]->tensor, model->weights.attn_v_b[i]->tensor);

        // reshape
        auto q_reshape = q_proj->tensor->reshape({ntoken, model->meta.nh, model->meta.dh});
        auto k_reshape = k_proj->tensor->reshape({ntoken, model->meta.nkvh, model->meta.dh});
        auto v_reshape = v_proj->tensor->reshape({ntoken, model->meta.nkvh, model->meta.dh});

        // RoPE
        llaisysTensor_t q_rope = createTensor(q_reshape->shape(), model->meta.dtype, model->device, device_id);
        llaisysTensor_t k_rope = createTensor(k_reshape->shape(), model->meta.dtype, model->device, device_id);

        llaisys::ops::rope(q_rope->tensor, q_reshape, position_ids_tensor->tensor, model->meta.theta);
        llaisys::ops::rope(k_rope->tensor, k_reshape, position_ids_tensor->tensor, model->meta.theta);

        auto k_cache_dst = model->k_cache[i]->tensor->slice(0, old_cached_len, new_cached_len);
        auto v_cache_dst = model->v_cache[i]->tensor->slice(0, old_cached_len, new_cached_len);
        copyTensor(k_cache_dst, k_rope->tensor);
        copyTensor(v_cache_dst, v_reshape);

        auto k_cache_total = model->k_cache[i]->tensor->slice(0, 0, new_cached_len);
        auto v_cache_total = model->v_cache[i]->tensor->slice(0, 0, new_cached_len);

        // self attention
        llaisysTensor_t attention_vals_raw = createTensor({ntoken, model->meta.nh, model->meta.dh}, model->meta.dtype, model->device, device_id);

        llaisys::ops::self_attention(attention_vals_raw->tensor, q_rope->tensor, k_cache_total, v_cache_total, 1.0 / std::sqrt(model->meta.dh));

        // reshape + linear
        llaisysTensor_t attention_vals = createTensor({ntoken, model->meta.hs}, model->meta.dtype, model->device, device_id);

        llaisys::ops::linear(attention_vals->tensor, attention_vals_raw->tensor->reshape({ntoken, q_dim}), model->weights.attn_o_w[i]->tensor, nullptr);

        // residual add
        llaisys::ops::add(attention_vals->tensor, attention_vals->tensor, attn_residual->tensor);

        /* MLP */
        llaisysTensor_t mlp_residual = attention_vals;

        // RNS NORM
        llaisys::ops::rms_norm(x_norm->tensor, attention_vals->tensor, model->weights.mlp_norm_w[i]->tensor, model->meta.epsilon);

        // projection
        llaisysTensor_t gate = createTensor({ntoken, model->meta.di}, model->meta.dtype, model->device, device_id);
        llaisysTensor_t up = createTensor({ntoken, model->meta.di}, model->meta.dtype, model->device, device_id);

        llaisys::ops::linear(gate->tensor, x_norm->tensor, model->weights.mlp_gate_w[i]->tensor, nullptr);
        llaisys::ops::linear(up->tensor, x_norm->tensor, model->weights.mlp_up_w[i]->tensor, nullptr);

        // SwiGLU
        llaisysTensor_t glu_tensor = createTensor(gate->tensor->shape(), model->meta.dtype, model->device, device_id);
        llaisys::ops::swiglu(glu_tensor->tensor, gate->tensor, up->tensor);

        // projection
        llaisysTensor_t mlp_out = createTensor({ntoken, model->meta.hs}, model->meta.dtype, model->device, device_id);
        llaisys::ops::linear(mlp_out->tensor, glu_tensor->tensor, model->weights.mlp_down_w[i]->tensor, nullptr);

        // residual add
        llaisys::ops::add(mlp_out->tensor, mlp_out->tensor, mlp_residual->tensor);
        current_tensor = mlp_out;

        destroyTensor(attn_residual);
        destroyTensor(x_norm);
        destroyTensor(q_proj);
        destroyTensor(k_proj);
        destroyTensor(v_proj);
        destroyTensor(q_rope);
        destroyTensor(k_rope);
        destroyTensor(attention_vals_raw);
        destroyTensor(mlp_residual);
        destroyTensor(gate);
        destroyTensor(up);
        destroyTensor(glu_tensor);
    }
    model->cached_len = new_cached_len;

    // RNS NORM
    llaisysTensor_t final_hidden = createTensor(current_tensor->tensor->shape(), model->meta.dtype, model->device, device_id);
    llaisys::ops::rms_norm(final_hidden->tensor, current_tensor->tensor, model->weights.out_norm_w->tensor, model->meta.epsilon);

    // logits
    auto last_hidden = final_hidden->tensor->slice(0, ntoken - 1, ntoken);
    llaisysTensor_t logits = createTensor({1, model->meta.voc}, model->meta.dtype, model->device, device_id);
    llaisys::ops::linear(logits->tensor, last_hidden, model->weights.out_embed->tensor, nullptr);

    // argmax
    llaisysTensor_t max_idx = createTensor({1}, LLAISYS_DTYPE_I64, model->device, device_id);
    llaisysTensor_t max_val = createTensor({1}, model->meta.dtype, model->device, device_id);
    llaisys::ops::argmax(max_idx->tensor, max_val->tensor, logits->tensor->reshape({model->meta.voc}));

    int64_t res;
    llaisys::core::context().setDevice(max_idx->tensor->deviceType(), max_idx->tensor->deviceId());
    llaisys::core::context().runtime().api()->device_synchronize();
    llaisys::core::context().runtime().api()->memcpy_sync(
        &res,
        max_idx->tensor->data(),
        sizeof(res),
        LLAISYS_MEMCPY_D2H);

    destroyTensor(input_tokens_tensor);
    destroyTensor(position_ids_tensor);
    destroyTensor(current_tensor);
    destroyTensor(final_hidden);
    destroyTensor(logits);
    destroyTensor(max_idx);
    destroyTensor(max_val);

    return res;
}

}
