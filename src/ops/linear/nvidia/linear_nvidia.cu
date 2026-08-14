#include "linear_nvidia.hpp"

#include "../../../utils.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace {
void checkCudaError(cudaError_t err, const char *api) {
    if (err != cudaSuccess) {
        std::cerr << "[ERROR] " << api << " failed: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error(api);
    }
}

void checkCublasError(cublasStatus_t status, const char *api) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[ERROR] " << api << " failed with cublas status " << static_cast<int>(status) << std::endl;
        throw std::runtime_error(api);
    }
}

template <typename T>
__device__ T add(T a, T b) {
    return a + b;
}

template <>
__device__ __half add(__half a, __half b) {
    return __hadd(a, b);
}

template <>
__device__ __nv_bfloat16 add(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __hadd(a, b);
}

template <typename T>
__global__ void addBiasKernel(T *out, const T *bias, size_t batch, size_t out_features) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t numel = batch * out_features;
    if (idx < numel) {
        size_t col = idx % out_features;
        out[idx] = add(out[idx], bias[col]);
    }
}

template <typename T>
void launchAddBias(std::byte *out, const std::byte *bias, size_t batch, size_t out_features) {
    if (!bias) {
        return;
    }

    constexpr int block_size = 256;
    size_t numel = batch * out_features;
    int grid_size = static_cast<int>((numel + block_size - 1) / block_size);

    addBiasKernel<<<grid_size, block_size>>>(reinterpret_cast<T *>(out), reinterpret_cast<const T *>(bias), batch,
                                             out_features);
    checkCudaError(cudaGetLastError(), "addBiasKernel");
}

template <typename T>
cudaDataType_t cudaDataType();

template <>
cudaDataType_t cudaDataType<__half>() {
    return CUDA_R_16F;
}

template <>
cudaDataType_t cudaDataType<__nv_bfloat16>() {
    return CUDA_R_16BF;
}

void launchLinearF32(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t batch,
                     size_t out_features, size_t in_features) {
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "cublasCreate");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    checkCublasError(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(out_features),
                                 static_cast<int>(batch), static_cast<int>(in_features), &alpha,
                                 reinterpret_cast<const float *>(weight), static_cast<int>(in_features),
                                 reinterpret_cast<const float *>(in), static_cast<int>(in_features), &beta,
                                 reinterpret_cast<float *>(out), static_cast<int>(out_features)),
                     "cublasSgemm");
    checkCublasError(cublasDestroy(handle), "cublasDestroy");

    launchAddBias<float>(out, bias, batch, out_features);
}

template <typename T>
void launchLinear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, size_t batch,
                  size_t out_features, size_t in_features) {
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "cublasCreate");

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cudaDataType_t dtype = cudaDataType<T>();

    checkCublasError(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(out_features),
                                  static_cast<int>(batch), static_cast<int>(in_features), &alpha, weight, dtype,
                                  static_cast<int>(in_features), in, dtype, static_cast<int>(in_features), &beta, out,
                                  dtype, static_cast<int>(out_features), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                     "cublasGemmEx");
    checkCublasError(cublasDestroy(handle), "cublasDestroy");

    launchAddBias<T>(out, bias, batch, out_features);
}
} // namespace

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch, size_t out_features, size_t in_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchLinearF32(out, in, weight, bias, batch, out_features, in_features);
    case LLAISYS_DTYPE_F16:
        return launchLinear<__half>(out, in, weight, bias, batch, out_features, in_features);
    case LLAISYS_DTYPE_BF16:
        return launchLinear<__nv_bfloat16>(out, in, weight, bias, batch, out_features, in_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
