#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(attn_val->shape().size() == 3 && q->shape().size() == 3 && k->shape().size() == 3 && v->shape().size() == 3,
                   "SelfAttention: expected all tensors to be 3D.");
    CHECK_ARGUMENT(attn_val->shape()[0] == q->shape()[0], "SelfAttention: output sequence length must match q.");
    CHECK_ARGUMENT(attn_val->shape()[1] == q->shape()[1], "SelfAttention: output heads must match q heads.");
    CHECK_ARGUMENT(attn_val->shape()[2] == v->shape()[2], "SelfAttention: output head dim must match v head dim.");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0], "SelfAttention: k and v sequence lengths must match.");
    CHECK_ARGUMENT(k->shape()[1] == v->shape()[1], "SelfAttention: k and v head counts must match.");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2], "SelfAttention: q and k head dims must match.");
    CHECK_ARGUMENT(q->shape()[1] % k->shape()[1] == 0, "SelfAttention: q heads must be divisible by kv heads.");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), q->shape()[0],
                                   k->shape()[0], q->shape()[1], k->shape()[1], q->shape()[2], v->shape()[2], scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
