
## About

This work addresses the aforementioned challenges by exploring integrating Approximate
Matrix Multiplication (AMM) algorithms into inference-serving frameworks
such as vLLM, presenting a novel approach to LLM inference optimization. vLLM is a fast and easy-to-use library for LLM inference and serving. To enhance
inference efficiency in vLLM deployments, this study explores the application
of approximate computation techniques within its transformer-based pipeline, directly
targeting the bottleneck introduced by matrix multiplication. Specifically, we investigate
the integration of pruning and hybrid AMM algorithms from the LibAMM
framework as a replacement for conventional dense matrix multiplications in the feedforward
and attention layers of vLLM. Additionally, we present a detailed evaluation
of experiments conducted to reduce inference latency using AMM, while analyzing
the resulting trade-offs in model accuracy, providing insights into the feasibility of
AMM-based optimizations for large-scale LLM inference.

Current inferencing optimizations of vLLM include

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), INT4, INT8, and FP8.
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
- Speculative decoding
- Chunked prefill

vLLM is chosen for this project as it is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor parallelism and pipeline parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
- Prefix caching support
- Multi-lora support


Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

