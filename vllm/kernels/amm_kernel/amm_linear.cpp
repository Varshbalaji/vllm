#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "amm_kernel.cuh"

namespace vllm {
namespace amm_linear {

torch::Tensor amm_linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias) 
{
    TORCH_CHECK(input.is_cuda(), "Input must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "Weight must be on CUDA");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias must be on CUDA");
    }

    const int batch_size = input.size(0);
    const int in_features = input.size(1);
    const int out_features = weight.size(0);

    auto output = torch::empty(
        {batch_size, out_features},
        input.options()
    );

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "amm_linear_forward", ([&] {
        launch_amm_gemm<scalar_t>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.has_value() ? bias.value().data_ptr<scalar_t>() : nullptr,
            batch_size,
            in_features,
            out_features,
            stream
        );
    }));

    return output;
}

TORCH_LIBRARY(amm_linear, m) {
    m.def("forward", amm_linear_forward);
}

} // namespace amm_linear
} // namespace vllm