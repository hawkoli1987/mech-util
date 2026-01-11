# GLM-4.6 vLLM Configuration Experiments

## Objective
Evaluate different vLLM configurations for serving GLM-4.6 MoE model on 8x H200 GPUs.

## Datasets
| Dataset | Samples | Prompt Tokens | Target Response | Total Seq Len |
|---------|---------|---------------|-----------------|---------------|
| 8k | 100 | ~6K | ~8K | ~14K |
| 64k | 100 | ~48K | ~64K | ~112K |

Location: `storage/samples/glm46_benchmark/`

## Experiment Matrix

### Group 1: Compilation & CUDA Graphs
| Exp | enforce-eager | CUDA Graphs | torch.compile | Dataset |
|-----|---------------|-------------|---------------|---------|
| 1A | yes (baseline) | disabled | disabled | 8k |
| 1B | no | enabled | enabled | 8k |

### Group 2: Expert Parallelism (EP)
| Exp | enable-expert-parallel | Dataset | Notes |
|-----|------------------------|---------|-------|
| 2A | yes (baseline) | 8k | Distributes MoE experts |
| 2B | no | 8k | Replicates all experts |

### Group 3: Distributed Context Parallel (DCP)
| Exp | context-parallel-size | Dataset | Notes |
|-----|----------------------|---------|-------|
| 3A | 1 (baseline) | 8k, 64k | No context parallelism |
| 3B | 2 | 64k | Splits context across 2 groups |
| 3C | 4 | 64k | Splits context across 4 groups |

### Group 4: Max Model Length
| Exp | max-model-len | Dataset | Notes |
|-----|---------------|---------|-------|
| 4A | 16384 | 8k | Short context |
| 4B | 32768 (baseline) | 8k | Medium context |
| 4C | 65536 | 8k, 64k | Long context |
| 4D | 131072 | 64k | Ultra-long context |

### Group 5: KV Cache Dtype
| Exp | kv-cache-dtype | Dataset | Notes |
|-----|----------------|---------|-------|
| 5A | fp8 (baseline) | 8k | 8-bit quantized KV cache |
| 5B | auto (bf16) | 8k | Full precision KV cache |

### Group 6: Prefix Caching
| Exp | enable-prefix-caching | Dataset | Notes |
|-----|----------------------|---------|-------|
| 6A | yes (baseline) | 8k | Caches common prefixes |
| 6B | no | 8k | No prefix caching |

### Group 7: Chunked Prefill
| Exp | enable-chunked-prefill | Dataset | Notes |
|-----|------------------------|---------|-------|
| 7A | yes (baseline) | 8k | Chunked prompt processing |
| 7B | no | 8k | Full prompt processing |

### Group 8: Self-Speculative Decoding (MTP)
| Exp | Speculative Decoding | num-spec-tokens | Dataset | Notes |
|-----|---------------------|-----------------|---------|-------|
| 8A | no (baseline) | - | 8k | Standard decoding |
| 8B | yes (ngram) | 2 | 8k | 2-token speculation |
| 8C | yes (ngram) | 4 | 8k | 4-token speculation |

## PBS Commands

### Group 1: Compilation
```bash
# 1A: Baseline (already running)
# 1B: Without enforce-eager
qsub -v 'EXP_NAME=1B_compiled,ENFORCE_EAGER=no,NUM_SAMPLES=100' experiments/exp_runner.pbs
```

### Group 2: Expert Parallelism
```bash
# 2B: Without EP
qsub -v 'EXP_NAME=2B_no_ep,ENABLE_EP=no,NUM_SAMPLES=100' experiments/exp_runner.pbs
```

### Group 3: DCP
```bash
# 3B: DCP with 2-way context parallel
qsub -v 'EXP_NAME=3B_dcp2,CONTEXT_PARALLEL_SIZE=2,MAX_MODEL_LEN=131072,DATASET_TYPE=64k,NUM_SAMPLES=100' experiments/exp_runner.pbs

# 3C: DCP with 4-way context parallel
qsub -v 'EXP_NAME=3C_dcp4,CONTEXT_PARALLEL_SIZE=4,MAX_MODEL_LEN=131072,DATASET_TYPE=64k,NUM_SAMPLES=100' experiments/exp_runner.pbs
```

### Group 4: Max Model Length
```bash
# 4A: Short context
qsub -v 'EXP_NAME=4A_len16k,MAX_MODEL_LEN=16384,NUM_SAMPLES=100' experiments/exp_runner.pbs

# 4C: Long context
qsub -v 'EXP_NAME=4C_len65k,MAX_MODEL_LEN=65536,NUM_SAMPLES=100' experiments/exp_runner.pbs

# 4D: Ultra-long context
qsub -v 'EXP_NAME=4D_len131k,MAX_MODEL_LEN=131072,DATASET_TYPE=64k,NUM_SAMPLES=100' experiments/exp_runner.pbs
```

### Group 5: KV Cache Dtype
```bash
# 5B: bf16 KV cache
qsub -v 'EXP_NAME=5B_kv_bf16,KV_CACHE_DTYPE=auto,NUM_SAMPLES=100' experiments/exp_runner.pbs
```

### Group 6: Prefix Caching
```bash
# 6B: No prefix caching
qsub -v 'EXP_NAME=6B_no_prefix_cache,ENABLE_PREFIX_CACHING=no,NUM_SAMPLES=100' experiments/exp_runner.pbs
```

### Group 7: Chunked Prefill
```bash
# 7B: No chunked prefill
qsub -v 'EXP_NAME=7B_no_chunked,ENABLE_CHUNKED_PREFILL=no,NUM_SAMPLES=100' experiments/exp_runner.pbs
```

### Group 8: MTP Speculative Decoding
```bash
# 8B: 2-token speculation
qsub -v 'EXP_NAME=8B_mtp2,ENABLE_MTP=yes,MTP_NUM_TOKENS=2,NUM_SAMPLES=100' experiments/exp_runner.pbs

# 8C: 4-token speculation
qsub -v 'EXP_NAME=8C_mtp4,ENABLE_MTP=yes,MTP_NUM_TOKENS=4,NUM_SAMPLES=100' experiments/exp_runner.pbs
```

## Results Summary

### Baseline Configuration (1A)
```bash
--tensor-parallel-size 8
--enable-expert-parallel
--enforce-eager
--dtype bfloat16
--kv-cache-dtype fp8
--gpu-memory-utilization 0.95
--max-model-len 32768
--enable-chunked-prefill
--enable-prefix-caching
--trust-remote-code
```

| Metric | Value |
|--------|-------|
| TPS | 8.69 |
| Latency | 58.9s |
| Model Memory | 82.6 GiB |
| Prompt Tokens | ~6100 |
| Completion Tokens | 512 (limited in initial tests) |

### Experiment Results Table

| Exp | Config | Dataset | TPS | Latency | Status | Notes |
|-----|--------|---------|-----|---------|--------|-------|
| 1A | baseline | 8k | 8.69 | 58.9s | ✅ | Initial baseline (512 tokens) |
| 1B | compiled | 8k | - | - | ⏳ | Pending rerun with 100 samples |
| 2B | no EP | 8k | - | - | ⏳ | Pending |
| 3B | DCP-2 | 64k | - | - | ⏳ | Pending |
| 3C | DCP-4 | 64k | - | - | ⏳ | Pending |
| 4A | len=16k | 8k | - | - | ⏳ | Pending |
| 4C | len=65k | 8k | - | - | ⏳ | Pending |
| 4D | len=131k | 64k | - | - | ⏳ | Pending |
| 5B | kv=bf16 | 8k | - | - | ⏳ | Pending |
| 6B | no prefix | 8k | - | - | ⏳ | Pending |
| 7B | no chunk | 8k | - | - | ⏳ | Pending |
| 8B | MTP-2 | 8k | - | - | ⏳ | Pending |
| 8C | MTP-4 | 8k | - | - | ⏳ | Pending |

## Configuration Parameter Reference

### Core Parameters
| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `--enforce-eager` | flag | - | Disables torch.compile and CUDA graphs |
| `--enable-expert-parallel` | flag | - | Enables EP for MoE models |
| `--context-parallel-size` | 1,2,4,8 | 1 | DCP: splits context across GPU groups |
| `--max-model-len` | int | model default | Maximum sequence length |
| `--kv-cache-dtype` | fp8,auto | auto | KV cache quantization |
| `--enable-prefix-caching` | flag | - | Cache common prompt prefixes |
| `--enable-chunked-prefill` | flag | - | Process prompts in chunks |

### Speculative Decoding
| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `--speculative-model` | model,[ngram] | - | Draft model for speculation |
| `--num-speculative-tokens` | 1-8 | - | Number of tokens to speculate |

### Memory Management
| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| `--gpu-memory-utilization` | 0.0-1.0 | 0.9 | Fraction of GPU memory to use |
| `--swap-space` | int (GB) | 4 | CPU swap space for KV cache |

## Notes

### max_num_seqs Auto-Calculation
The experiments do NOT set `--max-num-seqs` explicitly. vLLM will automatically calculate the optimal value based on:
- Available GPU memory after model loading
- max_model_len setting
- KV cache dtype (fp8 uses 50% less memory than bf16)

This ensures each configuration maximizes KV cache utilization.

### Distributed Context Parallel (DCP)
DCP splits long sequences across GPU groups for parallel processing:
- context-parallel-size=2: Sequence split across 2 GPU groups (4 GPUs each)
- context-parallel-size=4: Sequence split across 4 GPU groups (2 GPUs each)
- Best for very long sequences (64k+) where memory is the bottleneck
- May increase inter-GPU communication overhead

### MTP Speculative Decoding
GLM-4.6 may support Multi-Token Prediction heads for self-speculative decoding:
- Uses ngram-based speculation as a baseline
- num-speculative-tokens controls speculation depth
- Higher values = more aggressive speculation but higher rejection risk
