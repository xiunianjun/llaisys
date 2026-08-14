#include "../runtime_api.hpp"

#include <musa_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace llaisys::device::moore {

namespace runtime_api {
void checkCudaError(musaError_t err, const char *api) {
    if (err != musaSuccess) {
        std::cerr << "[ERROR] " << api << " failed: " << musaGetErrorString(err) << std::endl;
        throw std::runtime_error(api);
    }
}

musaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return musaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return musaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return musaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return musaMemcpyDeviceToDevice;
    default:
        CHECK_ARGUMENT(false, "invalid memcpy kind");
        return musaMemcpyDefault;
    }
}

int getDeviceCount() {
    int count = 0;
    checkCudaError(musaGetDeviceCount(&count), "musaGetDeviceCount");
    return count;
}

void setDevice(int device) {
    checkCudaError(musaSetDevice(device), "musaSetDevice");
}

void deviceSynchronize() {
    checkCudaError(musaDeviceSynchronize(), "musaDeviceSynchronize");
}

llaisysStream_t createStream() {
    musaStream_t stream;
    checkCudaError(musaStreamCreate(&stream), "musaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    checkCudaError(musaStreamDestroy(reinterpret_cast<musaStream_t>(stream)), "musaStreamDestroy");
}
void streamSynchronize(llaisysStream_t stream) {
    checkCudaError(musaStreamSynchronize(reinterpret_cast<musaStream_t>(stream)), "musaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    void *addr;
    checkCudaError(musaMalloc(&addr, size), "musaMalloc");
    return addr;
}

void freeDevice(void *ptr) {
    checkCudaError(musaFree(ptr), "musaFree");
}

void *mallocHost(size_t size) {
    void *addr;
    checkCudaError(musaHostAlloc(&addr, size, musaHostAllocDefault), "musaHostAlloc");
    return addr;
}

void freeHost(void *ptr) {
    checkCudaError(musaFreeHost(ptr), "musaFreeHost");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    checkCudaError(musaMemcpy(dst, src, size, toCudaMemcpyKind(kind)), "musaMemcpy");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    checkCudaError(musaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), reinterpret_cast<musaStream_t>(stream)), "musaMemcpyAsync");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::moore
