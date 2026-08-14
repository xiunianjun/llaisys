#include "linear_nvidia.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch, size_t out_features, size_t in_features) {
    TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops::nvidia
