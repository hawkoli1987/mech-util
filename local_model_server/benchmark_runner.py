#!/usr/bin/env python3
"""
Benchmark runner for GLM-4.6 vLLM configuration experiments.
Tests inference performance with different configurations.

Features:
- Uses full max_tokens from dataset (not limited)
- Collects detailed metrics (TTFT, TPS, memory usage)
- Supports streaming for accurate TTFT measurement
"""

import json
import time
import requests
import argparse
from typing import Dict, Any, List, Optional
import os
import sys


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load benchmark dataset from JSON file."""
    with open(dataset_path, "r") as f:
        return json.load(f)


def get_server_info(base_url: str) -> Optional[Dict]:
    """Get server model info including max_model_len."""
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get("data"):
                model = data["data"][0]
                return {
                    "model_id": model.get("id"),
                    "max_model_len": model.get("max_model_len"),
                }
        return None
    except Exception as e:
        print(f"Warning: Could not get server info: {e}")
        return None


def run_single_benchmark_streaming(
    base_url: str,
    model_name: str,
    messages: List[Dict],
    max_tokens: int,
    sample_id: str,
    timeout: int = 1800  # 30 minutes for long generations
) -> Dict[str, Any]:
    """Run a single benchmark with streaming to measure TTFT accurately."""
    
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True
    }
    
    start_time = time.time()
    first_token_time = None
    total_tokens = 0
    error_msg = None
    
    try:
        response = requests.post(url, json=payload, timeout=timeout, stream=True)
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            if chunk.get('choices') and chunk['choices'][0].get('delta', {}).get('content'):
                                if first_token_time is None:
                                    first_token_time = time.time()
                                # Count tokens (approximate by content length / 4)
                                content = chunk['choices'][0]['delta']['content']
                                total_tokens += max(1, len(content) // 4)
                        except json.JSONDecodeError:
                            pass
            
            total_time = time.time() - start_time
            ttft = first_token_time - start_time if first_token_time else total_time
            generation_time = total_time - ttft if first_token_time else total_time
            
            return {
                "sample_id": sample_id,
                "status": "success",
                "max_tokens_requested": max_tokens,
                "approx_completion_tokens": total_tokens,
                "total_time_s": round(total_time, 3),
                "ttft_s": round(ttft, 3),
                "generation_time_s": round(generation_time, 3),
                "tokens_per_second": round(total_tokens / generation_time, 2) if generation_time > 0 else 0,
                "error": None
            }
        else:
            error_msg = f"HTTP {response.status_code}: {response.text[:500]}"
    except requests.exceptions.Timeout:
        error_msg = f"Request timed out after {timeout}s"
    except Exception as e:
        error_msg = str(e)[:500]
    
    return {
        "sample_id": sample_id,
        "status": "error",
        "max_tokens_requested": max_tokens,
        "approx_completion_tokens": 0,
        "total_time_s": time.time() - start_time,
        "ttft_s": 0,
        "generation_time_s": 0,
        "tokens_per_second": 0,
        "error": error_msg
    }


def run_single_benchmark_non_streaming(
    base_url: str,
    model_name: str,
    messages: List[Dict],
    max_tokens: int,
    sample_id: str,
    timeout: int = 1800
) -> Dict[str, Any]:
    """Run a single benchmark without streaming for faster throughput testing."""
    
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        total_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            usage = result.get("usage", {})
            completion_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            
            return {
                "sample_id": sample_id,
                "status": "success",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "max_tokens_requested": max_tokens,
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
                "max_tokens_requested": max_tokens,
                "total_time_s": round(total_time, 3),
                "tokens_per_second": 0,
                "error": f"HTTP {response.status_code}: {response.text[:500]}"
            }
    except requests.exceptions.Timeout:
        return {
            "sample_id": sample_id,
            "status": "error",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "max_tokens_requested": max_tokens,
            "total_time_s": time.time() - start_time,
            "tokens_per_second": 0,
            "error": f"Request timed out after {timeout}s"
        }
    except Exception as e:
        return {
            "sample_id": sample_id,
            "status": "error",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "max_tokens_requested": max_tokens,
            "total_time_s": time.time() - start_time,
            "tokens_per_second": 0,
            "error": str(e)[:500]
        }


def run_benchmark_suite(
    base_url: str,
    model_name: str,
    dataset_path: str,
    num_samples: int = 100,
    config_name: str = "default",
    use_streaming: bool = False,
    max_tokens_override: Optional[int] = None
) -> Dict[str, Any]:
    """Run benchmark suite on a dataset."""
    
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Samples: {num_samples}")
    print(f"Streaming: {use_streaming}")
    if max_tokens_override:
        print(f"Max tokens override: {max_tokens_override}")
    print(f"{'='*60}")
    
    # Get server info
    server_info = get_server_info(base_url)
    if server_info:
        print(f"Server max_model_len: {server_info.get('max_model_len')}")
    
    dataset = load_dataset(dataset_path)[:num_samples]
    results = []
    
    for i, sample in enumerate(dataset):
        max_tokens = max_tokens_override if max_tokens_override else sample.get("max_tokens", 8192)
        
        # Cap max_tokens to server's max_model_len minus estimated prompt tokens
        if server_info and server_info.get("max_model_len"):
            # Rough estimate: assume prompt is about 6K-48K tokens based on dataset
            estimated_prompt = 6000 if "8k" in dataset_path else 48000
            available_tokens = server_info["max_model_len"] - estimated_prompt
            if max_tokens > available_tokens:
                max_tokens = max(512, available_tokens)
        
        print(f"  Running sample {i+1}/{num_samples} (max_tokens={max_tokens})...", end=" ", flush=True)
        
        if use_streaming:
            result = run_single_benchmark_streaming(
                base_url=base_url,
                model_name=model_name,
                messages=sample["messages"],
                max_tokens=max_tokens,
                sample_id=sample["id"]
            )
        else:
            result = run_single_benchmark_non_streaming(
                base_url=base_url,
                model_name=model_name,
                messages=sample["messages"],
                max_tokens=max_tokens,
                sample_id=sample["id"]
            )
        
        results.append(result)
        print(f"TPS: {result.get('tokens_per_second', 0)}, Status: {result['status']}")
        
        if result["status"] == "error":
            print(f"    Error: {result.get('error', 'Unknown')[:100]}")
    
    # Calculate aggregates
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        avg_tps = sum(r["tokens_per_second"] for r in successful) / len(successful)
        avg_time = sum(r["total_time_s"] for r in successful) / len(successful)
        total_tokens = sum(r.get("completion_tokens", r.get("approx_completion_tokens", 0)) for r in successful)
    else:
        avg_tps = 0
        avg_time = 0
        total_tokens = 0
    
    summary = {
        "config_name": config_name,
        "dataset": os.path.basename(dataset_path),
        "total_samples": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "avg_tokens_per_second": round(avg_tps, 2),
        "avg_time_s": round(avg_time, 3),
        "total_tokens_generated": total_tokens,
        "server_info": server_info,
        "results": results
    }
    
    print(f"\n  Summary: {len(successful)}/{len(results)} success")
    print(f"  Avg TPS: {avg_tps:.2f}")
    print(f"  Total tokens: {total_tokens}")
    
    return summary


def check_server_health(base_url: str) -> bool:
    """Check if the server is healthy."""
    try:
        response = requests.get(f"{base_url}/health", timeout=30)
        return response.status_code == 200
    except:
        return False


def main():
    parser = argparse.ArgumentParser(description="GLM-4.6 Benchmark Runner")
    parser.add_argument("--base-url", default="http://hopper-46:8002", help="vLLM server URL")
    parser.add_argument("--model", default="glm-4.6-coder", help="Model name")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to run")
    parser.add_argument("--config-name", default="default", help="Configuration name for logging")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--streaming", action="store_true", help="Use streaming for TTFT measurement")
    parser.add_argument("--max-tokens", type=int, help="Override max_tokens from dataset")
    
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
        config_name=args.config_name,
        use_streaming=args.streaming,
        max_tokens_override=args.max_tokens
    )
    
    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
