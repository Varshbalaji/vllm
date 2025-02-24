import argparse
import subprocess
import json

def run_lm_eval(model_path, output_file):
    """
    Runs lm-eval-harness to compute perplexity for the model running on vLLM.
    """
    command = [
        "lm-eval",
        "--model", "vllm",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
        "--tasks", "wikitext",
        "--device", "cuda:0",
        "--batch_size", "1",
        "--output_path", output_file
    ]
    
    print(f"Running command: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True)
        print(f"Results saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running lm-eval-harness: {e}")

def load_results(file_path):
    """
    Loads and prints the perplexity score from the evaluation results.
    """
    with open(file_path, "r") as f:
        results = json.load(f)
        perplexity = results.get("results", {}).get("wikitext", {}).get("perplexity", "N/A")
        print(f"Perplexity Score: {perplexity}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark perplexity for vLLM models")
    parser.add_argument("--quantized", action="store_true", help="Run with quantized vLLM model")
    
    args = parser.parse_args()

    if args.quantized:
        model_path = "/home/ubuntu/mistral7b_quantized"
        output_file = "/home/ubuntu/benchmark_results/quantized_perplexity.json"
    else:
        model_path = "/home/ubuntu/mistral7b"
        output_file = "/home/ubuntu/benchmark_results/non_quantized_perplexity.json"

    run_lm_eval(model_path, output_file)
    load_results(output_file)
