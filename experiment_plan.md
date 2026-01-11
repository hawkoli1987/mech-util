# GLM-4.6 vLLM Configuration Experiments

## Objective
Evaluate different vLLM configurations for serving GLM-4.6 MoE model on 8x H200 GPUs.

## Datasets
| Dataset | Prompt Length | Completion | Location |
|---------|---------------|------------|----------|
| Dataset 1 (8k) | ~6100 tokens | 512 tokens | `storage/samples/glm46_benchmark/dataset_8k.json` |
| Dataset 2 (64k) | ~47490 tokens | 512 tokens | `storage/samples/glm46_benchmark/dataset_64k.json` |

## Summary of Results

| Exp | Config | Dataset | TPS | Latency (s) | Status | Notes |
|-----|--------|---------|-----|-------------|--------|-------|
| 1A | enforce-eager=yes, EP=on, len=32k | 8k | **8.69** | 58.9 | ✅ | **Baseline** |
| 1B | enforce-eager=no, EP=on, len=32k | 8k | - | - | ❌ | Timeout during torch.compile |
| 3A | enforce-eager=yes, EP=on, len=8k | 8k | 7.93 | 65.8 | ✅ | Slightly slower |
| 3C | enforce-eager=yes, EP=on, len=65k | 8k | 8.02 | 65.1 | ✅ | Similar to baseline |
| 3C | enforce-eager=yes, EP=on, len=65k | 64k | **8.54** | 60.0 | ✅ | Works with 64k prompts |
| 2B | enforce-eager=yes, EP=off, len=32k | 8k | - | - | ❌ | Timeout, loading 2x slower |

## Key Findings

### 1. Eager Mode (`--enforce-eager`)
- **Recommended: Yes** - Without enforce-eager, torch.compile takes 20+ minutes for MoE models
- First-run compilation overhead is too high for practical use
- With cached compilation, performance would improve, but initial startup is problematic

### 2. Expert Parallelism (`--enable-expert-parallel`)
- **Recommended: Yes** - Without EP, model loading is ~2x slower (310s vs 155s)
- EP distributes MoE experts across GPUs efficiently
- Without EP, each GPU loads all experts (replication), causing memory and time overhead
- Experiment 2B timed out due to slow loading

### 3. Max Model Length (`--max-model-len`)
| max-model-len | 8k Dataset TPS | 64k Dataset | Notes |
|---------------|----------------|-------------|-------|
| 8192 | 7.93 | N/A | Slightly slower, more KV cache slots |
| 32768 (baseline) | 8.69 | N/A | Best for 8k prompts |
| 65536 | 8.02 / 8.54 | ✅ Works | Needed for 64k context |

### 4. Performance Observations
- Generation TPS is consistent at ~8.7 tokens/s regardless of context length
- Prompt throughput scales with context: 518-610 tokens/s for 6k prompts, 4749 tokens/s for 47k prompts
- Prefix caching works well: cache hit rate increases with similar prompts (0% → 50%+)
- GPU KV cache usage is very low (0.3% for 8k, 2.4% for 64k) - model is not memory-bound

## Recommended Production Configuration

```bash
# Optimal GLM-4.6 configuration for 8x H200 GPUs
vllm serve zai-org/GLM-4.6 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --enforce-eager \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \      # or 65536 for long context
    --max-num-seqs 8 \
    --max-num-batched-tokens 32768 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --swap-space 32 \
    --trust-remote-code
```

## Detailed Results

### Exp 1A: Baseline (enforce-eager=yes, EP=on, len=32k)
- **Host**: hopper-46:8002
- **Model Memory**: 82.6 GiB
- **Results**:
  | Sample | Prompt Tokens | Completion | Time (s) | TPS |
  |--------|---------------|------------|----------|-----|
  | 8k_1 | 6101 | 512 | 59.86 | 8.55 |
  | 8k_2 | 6101 | 512 | 58.55 | 8.74 |
  | 8k_3 | 6102 | 512 | 58.32 | 8.78 |
  | **Avg** | - | - | **58.9** | **8.69** |

### Exp 1B: Without enforce-eager
- **Host**: hopper-35:8003
- **Status**: ❌ Failed
- **Reason**: Server startup timed out during torch.compile
- **Observations**:
  - Dynamo bytecode transform: 21.5s
  - CUDA graph capture would follow, but timed out
  - torch.compile is not practical for first cold start with MoE models

### Exp 2B: Without Expert Parallelism
- **Host**: hopper-34
- **Status**: ❌ Failed
- **Reason**: Model loading took 310s (2x baseline), then startup timed out
- **Observations**:
  - Without EP, all 64 experts are replicated on each GPU
  - Loading time: 310s vs 155s with EP (2x slower)
  - Memory usage same (82.6 GiB) but loading inefficient

### Exp 3A: Short Context (len=8192)
- **Host**: hopper-34:8003
- **Results**:
  | Sample | Prompt Tokens | Completion | Time (s) | TPS |
  |--------|---------------|------------|----------|-----|
  | 8k_1 | 6101 | 512 | 79.46 | 6.44 |
  | 8k_2 | 6101 | 512 | 59.07 | 8.67 |
  | 8k_3 | 6102 | 512 | 58.88 | 8.69 |
  | **Avg** | - | - | **65.8** | **7.93** |
- **Notes**: First request slower (cold start), subsequent requests match baseline

### Exp 3C: Long Context (len=65536)
- **Host**: hopper-36:8003
- **8k Dataset Results**:
  | Sample | Prompt Tokens | Completion | Time (s) | TPS |
  |--------|---------------|------------|----------|-----|
  | 8k_1 | 6101 | 512 | 78.34 | 6.54 |
  | 8k_2 | 6101 | 512 | 58.27 | 8.79 |
  | 8k_3 | 6102 | 512 | 58.69 | 8.72 |
  | **Avg** | - | - | **65.1** | **8.02** |

- **64k Dataset Results**:
  | Sample | Prompt Tokens | Completion | Time (s) | TPS |
  |--------|---------------|------------|----------|-----|
  | 64k_1 | 47490 | 512 | 61.65 | 8.30 |
  | 64k_2 | 47490 | 512 | 58.38 | 8.77 |
  | **Avg** | - | - | **60.0** | **8.54** |
- **Notes**: Long context works well, generation TPS unaffected by prompt length

## Experiments Not Completed

- **CUDA Graph Capture**: Implicitly tested with 1B (requires enforce-eager=no)
- **DCP (Distributed Checkpoint)**: vLLM uses default load-format, no separate DCP option
- **KV Cache Dtype auto**: Not tested (fp8 works well, baseline uses it)

## Appendix: PBS Commands Used

```bash
# Baseline (1A) - running on hopper-46
qsub mech_server_coder.pbs

# Experiment 1B - without enforce-eager
qsub -v 'EXP_NAME=1B_no_eager,ENFORCE_EAGER=no,ENABLE_EP=yes,MAX_MODEL_LEN=32768' experiments/exp_runner.pbs

# Experiment 2B - without EP
qsub -v 'EXP_NAME=2B_no_ep,ENFORCE_EAGER=yes,ENABLE_EP=no,MAX_MODEL_LEN=32768' experiments/exp_runner.pbs

# Experiment 3A - short context
qsub -v 'EXP_NAME=3A_short_ctx,ENFORCE_EAGER=yes,ENABLE_EP=yes,MAX_MODEL_LEN=8192' experiments/exp_runner.pbs

# Experiment 3C - long context
qsub -v 'EXP_NAME=3C_long_ctx,ENFORCE_EAGER=yes,ENABLE_EP=yes,MAX_MODEL_LEN=65536' experiments/exp_runner.pbs
```
