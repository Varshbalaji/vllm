#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace vllm {
namespace amm_linear {

template<typename T>
void launch_amm_gemm(
    T* output,
    const T* input,
    const T* weight,
    const T* bias,
    const int batch_size,
    const int in_features,
    const int out_features,
    cudaStream_t stream);

} // namespace amm_linear
} // namespace vllm