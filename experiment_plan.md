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
| Max Seqs | `--max-num-seqs N` | 256 | **NOT auto-calculated** |
| Max Model Len | `--max-model-len N` | model default | Context window size |
| Spec Decoding | `--speculative-config JSON` | None | ngram, deepseek_mtp, etc. |

### ❌ NOT Supported in vLLM 0.11
| Feature | Status | Notes |
|---------|--------|-------|
| Context Parallel | Not available | `--context-parallel-size` doesn't exist |
| Sequence Parallel | Not available | No `--ulysses-sequence-parallel-size` |
| Ring Attention | Not available | Requires different architecture |

## Experiment Matrix (Verified)

### Group 1: Compilation & CUDA Graphs
| Exp | enforce-eager | Dataset | Status |
|-----|---------------|---------|--------|
| 1A | yes (baseline) | 8k | ✅ Baseline |
| 1B | no | 8k | ⏳ Pending (slow compile) |

### Group 2: Expert Parallelism (EP)
| Exp | enable-expert-parallel | Dataset | Status |
|-----|------------------------|---------|--------|
| 2A | yes (baseline) | 8k | ✅ Same as 1A |
| 2B | no | 8k | ❌ Timeout (slow load) |

### ~~Group 3: Distributed Context Parallel (DCP)~~
**CANCELLED** - vLLM 0.11 does not support context parallelism

### Group 4: Max Model Length
| Exp | max-model-len | Dataset | Status |
|-----|---------------|---------|--------|
| 4A | 16384 | 8k | ⏳ Running |
| 4B | 32768 (baseline) | 8k | ✅ Baseline |
| 4C | 65536 | 8k | ⏳ Running |

### Group 5: KV Cache Dtype
| Exp | kv-cache-dtype | Dataset | Status |
|-----|----------------|---------|--------|
| 5A | fp8 (baseline) | 8k | ✅ Baseline |
| 5B | auto (bf16) | 8k | ⏳ Running |

### Group 6: Prefix Caching
| Exp | enable-prefix-caching | Dataset | Status |
|-----|----------------------|---------|--------|
| 6A | yes (baseline) | 8k | ✅ Baseline |
| 6B | no | 8k | ⏳ Running |

### Group 7: Chunked Prefill
| Exp | chunked-prefill | Dataset | Status |
|-----|-----------------|---------|--------|
| 7A | yes (baseline) | 8k | ✅ Baseline |
| 7B | no | 8k | ⏳ Running (needs --no-enable-chunked-prefill) |

### Group 8: Speculative Decoding (ngram)
| Exp | speculative-config | Dataset | Status |
|-----|-------------------|---------|--------|
| 8A | none (baseline) | 8k | ✅ Baseline |
| 8B | ngram, 2 tokens | 8k | ❌ Failed (wrong args) |
| 8C | ngram, 4 tokens | 8k | ❌ Failed (wrong args) |

## Fixed PBS Commands

### Speculative Decoding (Correct Syntax)
```bash
# Use --speculative-config with JSON
qsub -v 'EXP_NAME=8B_spec2,ENABLE_MTP=yes,MTP_NUM_TOKENS=2,NUM_SAMPLES=100' experiments/exp_runner.pbs
```

### Disable Chunked Prefill (Correct Syntax)
```bash
# Use --no-enable-chunked-prefill to disable
qsub -v 'EXP_NAME=7B_no_chunked,ENABLE_CHUNKED_PREFILL=no,NUM_SAMPLES=100' experiments/exp_runner.pbs
```

## Important Findings

### 1. max_num_seqs is NOT Auto-Calculated
- vLLM uses default of 256 when not specified
- Original baseline used `--max-num-seqs 8` explicitly
- For fair comparison, either:
  - Use same max_num_seqs across experiments
  - Or let vLLM use default 256 for all

### 2. V1 Engine Defaults
The vLLM V1 engine (used in 0.11) has different defaults:
- `enable_chunked_prefill` defaults to **True** (was False in V0)
- Need `--no-enable-chunked-prefill` to disable

### 3. Speculative Decoding Syntax
vLLM 0.11 uses JSON config:
```bash
--speculative-config '{"method": "ngram", "num_speculative_tokens": 2, "prompt_lookup_max": 2}'
```

NOT the old syntax:
```bash
--num-speculative-tokens 2 --speculative-model [ngram]  # WRONG
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
    --max-num-seqs 8 \
    --max-num-batched-tokens 32768 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --swap-space 32 \
    --trust-remote-code
```

## Results Summary

| Exp | Config | TPS | Status | Notes |
|-----|--------|-----|--------|-------|
| 1A | baseline | 8.69 | ✅ | enforce-eager, EP, fp8 KV |
| 1B | compiled | - | ⏳ | torch.compile (slow) |
| 2B | no EP | - | ❌ | Load timeout (2x slower) |
| 4A | len=16k | - | ⏳ | - |
| 4C | len=65k | - | ⏳ | - |
| 5B | kv=bf16 | - | ⏳ | - |
| 6B | no prefix | - | ⏳ | - |
| 7B | no chunk | - | ⏳ | Needs script fix |
| 8B | spec-2 | - | ❌ | Wrong args |
| 8C | spec-4 | - | ❌ | Wrong args |
