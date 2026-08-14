#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.hpp"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/linear_moore.hpp"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(out->shape().size() == 2 && in->shape().size() == 2 && weight->shape().size() == 2,
                   "Linear: expected out, in, and weight to be 2D.");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "Linear: out rows must match input rows.");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0], "Linear: out columns must match weight rows.");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "Linear: input columns must match weight columns.");

    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        CHECK_ARGUMENT(bias->shape().size() == 1, "Linear: bias must be 1D.");
        CHECK_ARGUMENT(bias->shape()[0] == weight->shape()[0], "Linear: bias length must match weight rows.");
        ASSERT(bias->isContiguous(), "Linear: bias tensor must be contiguous.");
    }

    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "Linear: tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(),
                           out->shape()[0], out->shape()[1], in->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(),
                           out->shape()[0], out->shape()[1], in->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(),
                              out->shape()[0], out->shape()[1], in->shape()[1]);
#endif
#ifdef ENABLE_MOORE_API
    case LLAISYS_DEVICE_MOORE:
        return moore::linear(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, out->dtype(),
                             out->shape()[0], out->shape()[1], in->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
