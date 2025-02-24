// #include <cuda_runtime.h>
// #include <torch/extension.h>
// #include "LibAMM/src/AMM/CRS/CRS.hpp"
// #include "LibAMM/src/DataStructures/Matrix.hpp"
// #include "LibAMM/src/Utils/Config.hpp"

// namespace vllm {
// namespace amm_linear {

// template<typename T>
// class CUDAMatrix : public LibAMM::Matrix<T> {
// public:
//     CUDAMatrix(T* data, int rows, int cols) 
//         : data_(data), rows_(rows), cols_(cols) {}

//     T* getData() const override { return data_; }
//     int getRows() const override { return rows_; }
//     int getCols() const override { return cols_; }

// private:
//     T* data_;
//     int rows_;
//     int cols_;
// };

// // CUDA kernel for AMM computation
// template<typename T>
// __global__ void amm_kernel(
//     const T* __restrict__ input,
//     const T* __restrict__ weight,
//     T* __restrict__ output,
//     const int batch_size,
//     const int in_features,
//     const int out_features) 
// {
//     // Implement CRS algorithm logic here
//     // Following LibAMM's CRS.cpp implementation
//     // but adapted for CUDA execution
// }

// // Bias addition kernel
// template<typename T>
// __global__ void add_bias_kernel(
//     T* __restrict__ output,
//     const T* __restrict__ bias,
//     const int size) 
// {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < size) {
//         output[idx] += bias[idx % size];
//     }
// }

// template<typename T>
// void launch_amm_gemm(
//     T* output,
//     const T* input,
//     const T* weight,
//     const T* bias,
//     const int batch_size,
//     const int in_features,
//     const int out_features,
//     cudaStream_t stream) 
// {
//     // Create CUDA matrices using LibAMM's Matrix interface
//     CUDAMatrix<T> input_matrix(const_cast<T*>(input), batch_size, in_features);
//     CUDAMatrix<T> weight_matrix(const_cast<T*>(weight), out_features, in_features);
//     CUDAMatrix<T> output_matrix(output, batch_size, out_features);

//     // Configure CRS parameters
//     LibAMM::Config config;
//     config.aRow = batch_size;
//     config.aCol = in_features;
//     config.bCol = out_features;

//     // Initialize CRS algorithm
//     LibAMM::CRS<T> crs;
//     crs.paraseConfig(&config);

//     // Calculate grid and block dimensions
//     const int threads_per_block = 256;
//     const int num_blocks = (batch_size * out_features + threads_per_block - 1) / threads_per_block;

//     // Launch main AMM kernel
//     amm_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
//         input, weight, output,
//         batch_size, in_features, out_features);

//     // Add bias if provided
//     if (bias != nullptr) {
//         add_bias_kernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
//             output, bias, batch_size * out_features);
//     }
// }

// // Explicit instantiations
// template void launch_amm_gemm<float>(
//     float*, const float*, const float*, const float*,
//     const int, const int, const int, cudaStream_t);

// template void launch_amm_gemm<half>(
//     half*, const half*, const half*, const half*,
//     const int, const int, const int, cudaStream_t);

// } // namespace amm_linear
// } // namespace vll

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "amm_kernel.cuh"

namespace vllm {
namespace amm_linear {

// Helper kernel for initializing CURAND states
__global__ void setup_curand_kernel(curandState *state, unsigned long seed, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curand_init(seed, idx, 0, &state[idx]);
    }
}

// Kernel for sampling indices using multinomial distribution
__global__ void multinomial_sampling_kernel(
    curandState *state,
    int *sampled_indices,
    const float *probs,
    const int n,
    const int k) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k) {
        curandState localState = state[idx];
        float r = curand_uniform(&localState);
        float cumsum = 0.0f;
        
        // Simple multinomial sampling
        for (int i = 0; i < n; i++) {
            cumsum += probs[i];
            if (r <= cumsum) {
                sampled_indices[idx] = i;
                break;
            }
        }
        state[idx] = localState;
    }
}

// Kernel for matrix sampling and scaling
template<typename T>
__global__ void sample_and_scale_kernel(
    const T* input_matrix,
    T* sampled_matrix,
    const int* indices,
    const int n,
    const int k,
    const int dim)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < k && col < dim) {
        int orig_row = indices[row];
        sampled_matrix[row * dim + col] = input_matrix[orig_row * dim + col] * (T)(n) / (T)(k);
    }
}

// Main CRS GEMM kernel
template<typename T>
void launch_crs_gemm(
    T* output,
    const T* A,
    const T* B,
    const int batch_size,
    const int in_features,
    const int out_features,
    const int k,
    cudaStream_t stream)
{
    // Allocate memory for sampling
    curandState* d_states;
    int* d_indices;
    float* d_probs;
    T* d_A_sampled;
    T* d_B_sampled;
    
    cudaMalloc(&d_states, k * sizeof(curandState));
    cudaMalloc(&d_indices, k * sizeof(int));
    cudaMalloc(&d_probs, batch_size * sizeof(float));
    cudaMalloc(&d_A_sampled, k * in_features * sizeof(T));
    cudaMalloc(&d_B_sampled, k * out_features * sizeof(T));

    // Initialize uniform probabilities
    float prob_val = 1.0f / batch_size;
    cudaMemset(d_probs, prob_val, batch_size * sizeof(float));

    // Setup CURAND states
    int block_size = 256;
    int grid_size = (k + block_size - 1) / block_size;
    setup_curand_kernel<<<grid_size, block_size, 0, stream>>>(
        d_states, time(NULL), k);

    // Sample indices
    multinomial_sampling_kernel<<<grid_size, block_size, 0, stream>>>(
        d_states, d_indices, d_probs, batch_size, k);

    // Sample and scale matrices
    dim3 block_dims(16, 16);
    dim3 grid_dims(
        (in_features + block_dims.x - 1) / block_dims.x,
        (k + block_dims.y - 1) / block_dims.y
    );

    // Sample A matrix
    sample_and_scale_kernel<<<grid_dims, block_dims, 0, stream>>>(
        A, d_A_sampled, d_indices, batch_size, k, in_features);

    grid_dims.x = (out_features + block_dims.x - 1) / block_dims.x;
    
    // Sample B matrix
    sample_and_scale_kernel<<<grid_dims, block_dims, 0, stream>>>(
        B, d_B_sampled, d_indices, batch_size, k, out_features);

    // Perform final matrix multiplication using cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        out_features,    // m
        batch_size,      // n
        k,              // k
        &alpha,
        d_B_sampled,    // A
        out_features,    // lda
        d_A_sampled,    // B
        k,              // ldb
        &beta,
        output,         // C
        out_features    // ldc
    );

    // Cleanup
    cudaFree(d_states);
    cudaFree(d_indices);
    cudaFree(d_probs);
    cudaFree(d_A_sampled);
    cudaFree(d_B_sampled);
    cublasDestroy(handle);
}

// Launch function from header
template<typename T>
void launch_amm_gemm(
    T* output,
    const T* input,
    const T* weight,
    const T* bias,
    const int batch_size,
    const int in_features,
    const int out_features,
    cudaStream_t stream)
{
    // Use k=3 as shown in the notebook example
    const int k = 3;
    
    // Launch CRS implementation
    launch_crs_gemm(
        output,
        input,
        weight,
        batch_size,
        in_features,
        out_features,
        k,
        stream
    );

    // Add bias if provided
    if (bias != nullptr) {
        const int threads = 256;
        const int blocks = (batch_size * out_features + threads - 1) / threads;
        add_bias_kernel<<<blocks, threads, 0, stream>>>(
            output,
            bias,
            batch_size * out_features
        );
    }
}

// Explicit instantiation
template void launch_amm_gemm<float>(
    float*, const float*, const float*, const float*,
    const int, const int, const int, cudaStream_t);

} // namespace amm_linear
} // namespace vllm