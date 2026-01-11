# GLM-4.6 vLLM Configuration Experiments

## Objective
Evaluate different vLLM configurations for serving GLM-4.6 MoE model on 8x H200 GPUs.

## vLLM Version
- **vLLM 0.12.0** - Using `/scratch/Projects/SPEC-SF-AISG/source_files/Mech/4-sqsh/vllm-openai-v0.12.0.sqsh`

## Datasets
| Dataset | Samples | Prompt Tokens | Target Response | Total Seq Len |
|---------|---------|---------------|-----------------|---------------|
| 8k | 100 | ~6K | ~8K | ~14K |
| 64k | 100 | ~48K | ~64K | ~112K |

Location: `storage/samples/glm46_benchmark/`

## Two-Phase Experiment Methodology

### Why Two Phases?
vLLM does NOT auto-calculate `max_num_seqs` - it uses a default of 256. However, the optimal
value depends on the KV cache capacity, which varies with configuration (max_model_len, 
kv_cache_dtype, etc.). To ensure fair comparisons, we use:

**Phase 1: Probe Run**
- Start server with `max_num_seqs=1` (same configuration as Phase 2)
- vLLM calculates and logs the maximum concurrency during startup:
  ```
  Maximum concurrency for 32,768 tokens per request: 61.44x
  ```
- Extract this value from the logs (floor to integer → `max_num_seqs=61`)

**Phase 2: Actual Run**
- Restart server with the extracted `max_num_seqs`
- Run benchmark with 100 samples

This ensures each configuration uses its maximum possible concurrency based on vLLM's own
memory profiling, without manual calculation.

## Verified Configuration Parameters

### ✅ Supported in vLLM 0.12
| Parameter | Flag | Default | Notes |
|-----------|------|---------|-------|
| Eager Mode | `--enforce-eager` | False | Disables torch.compile & CUDA graphs |
| Expert Parallel | `--enable-expert-parallel` | False | Required for MoE models |
| Prefix Caching | `--enable-prefix-caching` | False | Caches common prefixes |
| Chunked Prefill | `--enable-chunked-prefill` | **True (V1)** | V1 engine defaults to True |
| Disable Chunked | `--no-enable-chunked-prefill` | - | Explicitly disable |
| KV Cache Type | `--kv-cache-dtype fp8\|auto` | auto | fp8 saves 50% memory |
| Max Seqs | `--max-num-seqs N` | 256 | **Extracted from vLLM logs in Phase 1** |
| Max Model Len | `--max-model-len N` | model default | Context window size |
| **DCP** | `--decode-context-parallel-size N` | 1 | Shards KV cache across GPUs |

### ✅ DCP (Decoder Context Parallelism) - NEW in vLLM 0.12
DCP shards the KV cache across multiple GPUs during the decoding phase:
- Increases total KV cache capacity
- Improves memory efficiency for long contexts
- Valid values: 1, 2, 4, 8 (must divide `tensor_parallel_size=8`)
- Best for 64k+ context workloads

### ❌ NOT Supported in vLLM 0.12
| Feature | Status | Notes |
|---------|--------|-------|
| Sequence Parallel | Not available | No `--ulysses-sequence-parallel-size` |
| Ring Attention | Not available | Requires different architecture |
| GLM-4.6 MTP | Not available | vLLM doesn't support GLM's `num_nextn_predict_layers` |

### ❌ MTP (Self-Speculative Decoding) NOT Supported for GLM-4.6
GLM-4.6 has `num_nextn_predict_layers=1` in its config, which is similar to DeepSeek's MTP.
However, vLLM 0.12 does NOT support GLM-specific MTP:
- `deepseek_mtp` method only works with DeepSeek models
- `ngram` method doesn't use the model's MTP heads (it's purely prompt-based)
- **Result**: MTP experiments removed from the test matrix

## Experiment Matrix (Final)

### Group 1: Compilation & CUDA Graphs
| Exp | enforce-eager | Dataset | Status | Notes |
|-----|---------------|---------|--------|-------|
| 1A | yes (baseline) | 8k | ✅ Baseline | Fast cold start |
| 1B | no | 8k | ❌ Timeout | torch.compile too slow for MoE |

### Group 2: Expert Parallelism (EP)
| Exp | enable-expert-parallel | Dataset | Status | Notes |
|-----|------------------------|---------|--------|-------|
| 2A | yes (baseline) | 8k | ✅ Same as 1A | Required for GLM-4.6 |
| 2B | no | 8k | ❌ Timeout | 2x slower load, likely OOM |

### Group 3: Max Model Length
| Exp | max-model-len | Dataset | Status | Notes |
|-----|---------------|---------|--------|-------|
| 3A | 16384 | 8k | ⏳ Pending | Lower memory, higher max_num_seqs |
| 3B | 32768 (baseline) | 8k | ✅ Baseline | - |
| 3C | 65536 | 8k | ⏳ Pending | Higher memory, lower max_num_seqs |

### Group 4: KV Cache Dtype
| Exp | kv-cache-dtype | Dataset | Status | Notes |
|-----|----------------|---------|--------|-------|
| 4A | fp8 (baseline) | 8k | ✅ Baseline | 50% KV memory savings |
| 4B | auto (bf16) | 8k | ⏳ Pending | 2x KV memory, lower max_num_seqs |

### Group 5: Prefix Caching
| Exp | enable-prefix-caching | Dataset | Status | Notes |
|-----|----------------------|---------|--------|-------|
| 5A | yes (baseline) | 8k | ✅ Baseline | Good for repeated prompts |
| 5B | no | 8k | ⏳ Pending | - |

### Group 6: Chunked Prefill
| Exp | chunked-prefill | Dataset | Status | Notes |
|-----|-----------------|---------|--------|-------|
| 6A | yes (baseline) | 8k | ✅ Baseline | V1 default |
| 6B | no | 8k | ⏳ Pending | Uses `--no-enable-chunked-prefill` |

### Group 7: Decoder Context Parallelism (DCP)
| Exp | decode-context-parallel-size | max-model-len | Dataset | Status | Notes |
|-----|------------------------------|---------------|---------|--------|-------|
| 7A | 1 (baseline) | 65536 | 64k | ⏳ Pending | No DCP, long context |
| 7B | 2 | 65536 | 64k | ⏳ Pending | DCP=2, KV sharded across 2 GPUs |
| 7C | 4 | 65536 | 64k | ⏳ Pending | DCP=4, KV sharded across 4 GPUs |

## PBS Submission Commands

```bash
cd /scratch/Projects/SPEC-SF-AISG/source_files/Mech/mech-util/local_model_server

# Group 3: Max Model Length
qsub -v 'EXP_NAME=3A_len16k,MAX_MODEL_LEN=16384' experiments/exp_runner.pbs
qsub -v 'EXP_NAME=3C_len65k,MAX_MODEL_LEN=65536' experiments/exp_runner.pbs

# Group 4: KV Cache
qsub -v 'EXP_NAME=4B_kv_bf16,KV_CACHE_DTYPE=auto' experiments/exp_runner.pbs

# Group 5: Prefix Caching
qsub -v 'EXP_NAME=5B_no_prefix,ENABLE_PREFIX_CACHING=no' experiments/exp_runner.pbs

# Group 6: Chunked Prefill
qsub -v 'EXP_NAME=6B_no_chunked,ENABLE_CHUNKED_PREFILL=no' experiments/exp_runner.pbs

# Group 7: DCP (use 64k dataset for long context)
qsub -v 'EXP_NAME=7A_dcp1_64k,MAX_MODEL_LEN=65536,DECODE_CONTEXT_PARALLEL_SIZE=1,DATASET_TYPE=64k' experiments/exp_runner.pbs
qsub -v 'EXP_NAME=7B_dcp2_64k,MAX_MODEL_LEN=65536,DECODE_CONTEXT_PARALLEL_SIZE=2,DATASET_TYPE=64k' experiments/exp_runner.pbs
qsub -v 'EXP_NAME=7C_dcp4_64k,MAX_MODEL_LEN=65536,DECODE_CONTEXT_PARALLEL_SIZE=4,DATASET_TYPE=64k' experiments/exp_runner.pbs
```

## Baseline Configuration
```bash
vllm serve zai-org/GLM-4.6 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --enforce-eager \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --max-num-seqs <extracted-from-vllm-logs> \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --swap-space 32 \
    --trust-remote-code
```

## Results Summary

| Exp | Config | max_num_seqs | TPS | Status | Notes |
|-----|--------|--------------|-----|--------|-------|
| 1A | baseline | 61 | 8.69 | ✅ | vLLM reported 61.44x concurrency |
| 1B | compiled | - | - | ❌ | torch.compile timeout |
| 2B | no EP | - | - | ❌ | Load timeout (2x slower) |
| 3A | len=16k | TBD | - | ⏳ | Higher concurrency expected |
| 3C | len=65k | TBD | - | ⏳ | Lower concurrency expected |
| 4B | kv=bf16 | TBD | - | ⏳ | 2x KV memory |
| 5B | no prefix | TBD | - | ⏳ | - |
| 6B | no chunk | TBD | - | ⏳ | - |
| 7A | dcp1,len=65k | TBD | - | ⏳ | Baseline for DCP comparison |
| 7B | dcp2,len=65k | TBD | - | ⏳ | KV sharded across 2 GPUs |
| 7C | dcp4,len=65k | TBD | - | ⏳ | KV sharded across 4 GPUs |

## Key Findings

### 1. `enforce-eager` is Required
- Without it, `torch.compile` takes 20+ minutes for GLM-4.6 MoE model
- Cold start time is critical for HPC job scheduling

### 2. Expert Parallelism is Critical
- Disabling EP doubles model load time (155s → 310s)
- Likely causes OOM or severe performance degradation

### 3. MTP Not Available
- GLM-4.6 has MTP layers but vLLM 0.12 doesn't support GLM-specific MTP
- ngram speculation doesn't use model's MTP heads

### 4. Dynamic max_num_seqs via vLLM Logs
- vLLM logs `Maximum concurrency for X tokens per request: Y.YYx` during startup
- Extract Y.YY, floor to integer for optimal `max_num_seqs`
- Each configuration has different KV cache capacity → different optimal concurrency

### 5. DCP for Long Contexts
- DCP shards KV cache across GPUs, increasing total capacity
- Best used with long contexts (64k+) where single-GPU KV cache is insufficient
- Must divide tensor_parallel_size (8): valid values are 1, 2, 4, 8
