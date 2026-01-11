#!/usr/bin/env python3
"""
Benchmark runner for GLM-4.6 vLLM configuration experiments.
Tests inference performance with different configurations.
"""

import json
import time
import requests
import argparse
from typing import Dict, Any, List
import os

def load_dataset(dataset_path: str) -> List[Dict]:
    """Load benchmark dataset from JSON file."""
    with open(dataset_path, "r") as f:
        return json.load(f)


def run_single_benchmark(
    base_url: str,
    model_name: str,
    messages: List[Dict],
    max_tokens: int,
    sample_id: str
) -> Dict[str, Any]:
    """Run a single benchmark request and measure performance."""
    
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": min(max_tokens, 512),  # Limit for quick benchmarks
        "temperature": 0.7,
        "stream": False
    }
    
    start_time = time.time()
    first_token_time = None
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
            prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
            
            return {
                "sample_id": sample_id,
                "status": "success",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_time_s": round(total_time, 3),
                "tokens_per_second": round(completion_tokens / total_time, 2) if total_time > 0 else 0,
                "error": None
            }
        else:
            return {
                "sample_id": sample_id,
                "status": "error",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_time_s": round(total_time, 3),
                "tokens_per_second": 0,
                "error": f"HTTP {response.status_code}: {response.text[:200]}"
            }
    except Exception as e:
        return {
            "sample_id": sample_id,
            "status": "error",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_time_s": time.time() - start_time,
            "tokens_per_second": 0,
            "error": str(e)
        }


def run_benchmark_suite(
    base_url: str,
    model_name: str,
    dataset_path: str,
    num_samples: int = 3,
    config_name: str = "default"
) -> Dict[str, Any]:
    """Run benchmark suite on a dataset."""
    
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Samples: {num_samples}")
    print(f"{'='*60}")
    
    dataset = load_dataset(dataset_path)[:num_samples]
    results = []
    
    for i, sample in enumerate(dataset):
        print(f"  Running sample {i+1}/{num_samples}...", end=" ", flush=True)
        result = run_single_benchmark(
            base_url=base_url,
            model_name=model_name,
            messages=sample["messages"],
            max_tokens=sample.get("max_tokens", 512),
            sample_id=sample["id"]
        )
        results.append(result)
        print(f"TPS: {result['tokens_per_second']}, Status: {result['status']}")
    
    # Calculate aggregates
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        avg_tps = sum(r["tokens_per_second"] for r in successful) / len(successful)
        avg_time = sum(r["total_time_s"] for r in successful) / len(successful)
    else:
        avg_tps = 0
        avg_time = 0
    
    summary = {
        "config_name": config_name,
        "dataset": os.path.basename(dataset_path),
        "total_samples": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "avg_tokens_per_second": round(avg_tps, 2),
        "avg_time_s": round(avg_time, 3),
        "results": results
    }
    
    print(f"\n  Summary: {len(successful)}/{len(results)} success, Avg TPS: {avg_tps:.2f}")
    
    return summary


def check_server_health(base_url: str) -> bool:
    """Check if the server is healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        return response.status_code == 200
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description="GLM-4.6 Benchmark Runner")
    parser.add_argument("--base-url", default="http://hopper-46:8002", help="vLLM server URL")
    parser.add_argument("--model", default="glm-4.6-coder", help="Model name")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples to run")
    parser.add_argument("--config-name", default="default", help="Configuration name for logging")
    parser.add_argument("--output", help="Output JSON file for results")
    
    args = parser.parse_args()
    
    # Check server health
    if not check_server_health(args.base_url):
        print(f"ERROR: Server at {args.base_url} is not healthy")
        return 1
    
    print(f"Server at {args.base_url} is healthy")
    
    # Run benchmark
    results = run_benchmark_suite(
        base_url=args.base_url,
        model_name=args.model,
        dataset_path=args.dataset,
        num_samples=args.samples,
        config_name=args.config_name
    )
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    exit(main())
