#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace llaisys::device::nvidia {

namespace runtime_api {
void checkCudaError(cudaError_t err, const char *api) {
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] " << api << " failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(api);
    }
}

cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        CHECK_ARGUMENT(false, "invalid memcpy kind");
        return cudaMemcpyDefault;
    }
}

int getDeviceCount() {
    int count = 0;
    checkCudaError(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
}

void setDevice(int device) {
    checkCudaError(cudaSetDevice(device), "cudaSetDevice");
}

void deviceSynchronize() {
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "cudaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    checkCudaError(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamDestroy");
}
void streamSynchronize(llaisysStream_t stream) {
    checkCudaError(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), "cudaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    void *addr;
    checkCudaError(cudaMalloc(&addr, size), "cudaMalloc");
    return addr;
}

void freeDevice(void *ptr) {
    checkCudaError(cudaFree(ptr), "cudaFree");
}

void *mallocHost(size_t size) {
    void *addr;
    checkCudaError(cudaHostAlloc(&addr, size, cudaHostAllocDefault), "cudaHostAlloc");
    return addr;
}

void freeHost(void *ptr) {
    checkCudaError(cudaFreeHost(ptr), "cudaFreeHost");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    checkCudaError(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)), "cudaMemcpy");
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    checkCudaError(cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), reinterpret_cast<cudaStream_t>(stream)), "cudaMemcpyAsync");
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
} // namespace llaisys::device::nvidia
