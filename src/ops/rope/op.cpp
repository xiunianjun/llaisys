#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.hpp"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/rope_moore.hpp"
#endif

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64.");
    CHECK_ARGUMENT(out->shape().size() == 3 && in->shape().size() == 3 && pos_ids->shape().size() == 1,
                   "RoPE: expected out and in to be 3D, and pos_ids to be 1D.");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(pos_ids->shape()[0] == in->shape()[0], "RoPE: pos_ids length must match sequence length.");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0, "RoPE: head dimension must be even.");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(), "RoPE: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), out->shape()[0], out->shape()[1],
                         out->shape()[2], theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), out->shape()[0], out->shape()[1],
                         out->shape()[2], theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), out->shape()[0], out->shape()[1],
                            out->shape()[2], theta);
#endif
#ifdef ENABLE_MOORE_API
    case LLAISYS_DEVICE_MOORE:
        return moore::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), out->shape()[0], out->shape()[1],
                           out->shape()[2], theta);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
