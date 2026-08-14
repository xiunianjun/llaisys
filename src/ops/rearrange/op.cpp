#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rearrange_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rearrange_nvidia.hpp"
#endif

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(out->numel() == in->numel(), "Rearrange: input and output must have the same number of elements.");
    ASSERT(out->isContiguous() && in->isContiguous(), "Rearrange: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rearrange(out->data(), in->data(), out->dtype(), out->numel());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rearrange(out->data(), in->data(), out->dtype(), out->numel());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
