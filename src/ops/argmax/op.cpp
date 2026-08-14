#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.hpp"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/argmax_moore.hpp"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be int64.");
    CHECK_ARGUMENT(max_idx->numel() == 1, "Argmax: max_idx must contain exactly one element.");
    CHECK_ARGUMENT(max_val->numel() == 1, "Argmax: max_val must contain exactly one element.");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "Argmax: all tensors must be contiguous.");

    // always support cpu calculation
    if (max_idx->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    llaisys::core::context().setDevice(max_idx->deviceType(), max_idx->deviceId());

    switch (max_idx->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#endif
#ifdef ENABLE_MOORE_API
    case LLAISYS_DEVICE_MOORE:
        return moore::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
