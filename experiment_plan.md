# GLM-4.6 vLLM Configuration Experiments

## Objective
Evaluate different vLLM configurations for serving GLM-4.6 MoE model on 8x H200 GPUs.

## vLLM Version
- **vLLM 0.11.2** - Verified configuration options below

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
- Start server with `max_num_seqs=1` to minimize memory overhead
- Extract GPU block count from vLLM startup logs
- Calculate optimal `max_num_seqs = GPU_blocks / (max_model_len / block_size)`

**Phase 2: Actual Run**
- Restart server with calculated optimal `max_num_seqs`
- Run benchmark with 100 samples

This ensures each configuration uses its maximum possible concurrency.

## Verified Configuration Parameters

### ✅ Supported in vLLM 0.11
| Parameter | Flag | Default | Notes |
|-----------|------|---------|-------|
| Eager Mode | `--enforce-eager` | False | Disables torch.compile & CUDA graphs |
| Expert Parallel | `--enable-expert-parallel` | False | Required for MoE models |
| Prefix Caching | `--enable-prefix-caching` | False | Caches common prefixes |
| Chunked Prefill | `--enable-chunked-prefill` | **True (V1)** | V1 engine defaults to True |
| Disable Chunked | `--no-enable-chunked-prefill` | - | Explicitly disable |
| KV Cache Type | `--kv-cache-dtype fp8\|auto` | auto | fp8 saves 50% memory |
| Max Seqs | `--max-num-seqs N` | 256 | **Dynamically calculated in Phase 1** |
| Max Model Len | `--max-model-len N` | model default | Context window size |

### ❌ NOT Supported in vLLM 0.11
| Feature | Status | Notes |
|---------|--------|-------|
| Context Parallel | Not available | `--context-parallel-size` doesn't exist |
| Sequence Parallel | Not available | No `--ulysses-sequence-parallel-size` |
| Ring Attention | Not available | Requires different architecture |
| GLM-4.6 MTP | Not available | vLLM doesn't support GLM's `num_nextn_predict_layers` |

### ❌ MTP (Self-Speculative Decoding) NOT Supported for GLM-4.6
GLM-4.6 has `num_nextn_predict_layers=1` in its config, which is similar to DeepSeek's MTP.
However, vLLM 0.11 does NOT support GLM-specific MTP:
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
| 4A | 16384 | 8k | ⏳ Pending | Lower memory, higher max_num_seqs |
| 4B | 32768 (baseline) | 8k | ✅ Baseline | - |
| 4C | 65536 | 8k | ⏳ Pending | Higher memory, lower max_num_seqs |

### Group 4: KV Cache Dtype
| Exp | kv-cache-dtype | Dataset | Status | Notes |
|-----|----------------|---------|--------|-------|
| 5A | fp8 (baseline) | 8k | ✅ Baseline | 50% KV memory savings |
| 5B | auto (bf16) | 8k | ⏳ Pending | 2x KV memory, lower max_num_seqs |

### Group 5: Prefix Caching
| Exp | enable-prefix-caching | Dataset | Status | Notes |
|-----|----------------------|---------|--------|-------|
| 6A | yes (baseline) | 8k | ✅ Baseline | Good for repeated prompts |
| 6B | no | 8k | ⏳ Pending | - |

### Group 6: Chunked Prefill
| Exp | chunked-prefill | Dataset | Status | Notes |
|-----|-----------------|---------|--------|-------|
| 7A | yes (baseline) | 8k | ✅ Baseline | V1 default |
| 7B | no | 8k | ⏳ Pending | Uses `--no-enable-chunked-prefill` |

## PBS Submission Commands

```bash
cd /scratch/Projects/SPEC-SF-AISG/source_files/Mech/mech-util/local_model_server

# Group 3: Max Model Length
qsub -v 'EXP_NAME=4A_len16k,MAX_MODEL_LEN=16384' experiments/exp_runner.pbs
qsub -v 'EXP_NAME=4C_len65k,MAX_MODEL_LEN=65536' experiments/exp_runner.pbs

# Group 4: KV Cache
qsub -v 'EXP_NAME=5B_kv_bf16,KV_CACHE_DTYPE=auto' experiments/exp_runner.pbs

# Group 5: Prefix Caching
qsub -v 'EXP_NAME=6B_no_prefix,ENABLE_PREFIX_CACHING=no' experiments/exp_runner.pbs

# Group 6: Chunked Prefill
qsub -v 'EXP_NAME=7B_no_chunked,ENABLE_CHUNKED_PREFILL=no' experiments/exp_runner.pbs
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
    --max-num-seqs <dynamically-calculated> \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --swap-space 32 \
    --trust-remote-code
```

## Results Summary

| Exp | Config | max_num_seqs | TPS | Status | Notes |
|-----|--------|--------------|-----|--------|-------|
| 1A | baseline | 8 | 8.69 | ✅ | enforce-eager, EP, fp8 KV |
| 1B | compiled | - | - | ❌ | torch.compile timeout |
| 2B | no EP | - | - | ❌ | Load timeout (2x slower) |
| 4A | len=16k | TBD | - | ⏳ | Higher concurrency expected |
| 4C | len=65k | TBD | - | ⏳ | Lower concurrency expected |
| 5B | kv=bf16 | TBD | - | ⏳ | 2x KV memory |
| 6B | no prefix | TBD | - | ⏳ | - |
| 7B | no chunk | TBD | - | ⏳ | - |

## Key Findings

### 1. `enforce-eager` is Required
- Without it, `torch.compile` takes 20+ minutes for GLM-4.6 MoE model
- Cold start time is critical for HPC job scheduling

### 2. Expert Parallelism is Critical
- Disabling EP doubles model load time (155s → 310s)
- Likely causes OOM or severe performance degradation

### 3. MTP Not Available
- GLM-4.6 has MTP layers but vLLM 0.11 doesn't support GLM-specific MTP
- ngram speculation doesn't use model's MTP heads

### 4. Dynamic max_num_seqs
- Each configuration has different KV cache capacity
- Two-phase approach ensures optimal concurrency per config
