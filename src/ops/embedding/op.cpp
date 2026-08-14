#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.hpp"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be int64.");
    CHECK_ARGUMENT(out->shape().size() == 2 && index->shape().size() == 1 && weight->shape().size() == 2,
                   "Embedding: expected out to be 2D, index to be 1D, and weight to be 2D.");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0], "Embedding: out rows must match index length.");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[1], "Embedding: out columns must match weight columns.");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), index->numel(), weight->shape()[0],
                              weight->shape()[1] * weight->elementSize());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), index->numel(), weight->shape()[0],
                                 weight->shape()[1] * weight->elementSize());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
