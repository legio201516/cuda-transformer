# cuda-transformer

CUDA/C++ GPU training from scratch — no PyTorch, no ML frameworks.  
MLP done. Transformer in progress.

## MLP — MNIST, 40 epochs, RTX 4050 (CC 8.9)

| | Time | vs torch.compile |
|---|---|---|
| `torch.compile` | 2.5s | 1× |
| `mlp_bf16_fwd.cu` | 1.7s | 1.5× |
| **`mlp_bf16_full.cu`** | **0.56s** | **4.4×** |

## What's in the MLP

- `cublasGemmEx` + `CUBLAS_COMPUTE_32F_FAST_16F` (Tensor Cores)
- Full dataset on GPU — zero HtoD in training loop  
- Fused bias+ReLU, fused weight update (1 kernel launch for all params)
- Warp softmax via `__shfl_down_sync` — no shared memory needed
- Mixed precision: FP32 master weights, BF16 compute

## Next — transformer.cu

GPT decoder on TinyShakespeare, same constraints (pure CUDA, beat torch.compile).
[token + pos embed] → × N layers ( LayerNorm → Attention → LayerNorm → FFN ) → loss
Kernels to write: `gelu.cuh`, `layernorm.cuh`, `attention.cuh` (targeting FlashAttention-style)  

Kernels reused from MLP: warp softmax, BF16 GemmEx, fused update

## Build

```bash
nvcc -O3 -arch=sm_89 src/mlp_bf16_full.cu -lcublas -o mlp && ./mlp
```

## Profile

```bash
python3 nsysrun.py run <suffix> ./mlp
# → paul_implem/profiles/<date>_<suffix>_stats.txt
```

