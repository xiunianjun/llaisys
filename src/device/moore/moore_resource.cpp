#include "moore_resource.cuh"

namespace llaisys::device::moore {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_MOORE, device_id) {}

} // namespace llaisys::device::moore
