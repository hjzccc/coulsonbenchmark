import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.cuda.set_device(device)
torch.set_default_device(device)

def is_blackwell():
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 10

HAS_FP4 = False

if is_blackwell():
    try:
        import torch
        import transformer_engine.pytorch as te
        import transformer_engine_torch as tex
        from transformer_engine.common import recipe
        from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer, NVFP4Tensor
        from transformer_engine.pytorch.cpp_extensions import general_gemm
        from transformer_engine.pytorch.module.base import get_workspace
        HAS_FP4 = True
        print("FP4 support: ENABLED (Blackwell GPU detected)")
    except (ImportError, AttributeError) as e:
        print(f"FP4 support: DISABLED ({e})")
else:
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    print(f"FP4 support: DISABLED (requires Blackwell SM100+, current: SM{cap[0]}{cap[1]})")

ACCUM_DTYPE = torch.bfloat16
batch_sizes = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096,8192, 16384, 32768, 65536, 131072, 262144])
num_repeats = 10
num_warmups = 2
num_graph_repeats = 100


def align_to(x, alignment):
    return ((x + alignment - 1) // alignment) * alignment


@torch.inference_mode()
def benchmark_bf16_matmul(N, K):
    """Pure BF16 matmul benchmark."""
    results = []
    
    for m in batch_sizes:
        m_aligned = align_to(m, 16)
        
        A = torch.randn(m_aligned, K, device=device, dtype=ACCUM_DTYPE).contiguous()
        B = torch.randn(K, N, device=device, dtype=ACCUM_DTYPE).contiguous()
        
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.graph(graph,stream=stream):
            for _ in range(num_graph_repeats):
                C = torch.matmul(A, B)
        # Warm-up
        for _ in range(num_warmups):
            graph.replay()
        
        torch.cuda.synchronize()
        times = []
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        results.append(np.mean(times))
    
    return results


@torch.inference_mode()
def benchmark_fp4_general_gemm(N, K):
    """
    FP4 GEMM using general_gemm with pre-quantized weights.
    Activations are quantized on-the-fly (realistic inference).
    """
    if not HAS_FP4:
        return [0.0] * len(batch_sizes)
    
    results = []
    
    K_aligned = align_to(K, 16)
    N_aligned = align_to(N, 16)
    
    # Weight quantizer: no RHT, 2D quantization for weights
    weight_quantizer = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_rht=False,
        with_2d_quantization=True,
        stochastic_rounding=False,
    )
    
    # Pre-quantize weights [K, N] (weights are stored as [out_features, in_features])
    # For GEMM: y = x @ W^T, so W is [N, K] and we compute [M, K] @ [K, N] = [M, N]
    W_bf16 = torch.randn(N_aligned, K_aligned, device=device, dtype=ACCUM_DTYPE).contiguous()
    W_fp4 = weight_quantizer.quantize_impl(W_bf16)
    
    # Activation quantizer: with RHT for activations
    act_quantizer = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=False,
        with_rht=True,
        with_post_rht_amax=True,
        with_2d_quantization=False,
        stochastic_rounding=False,
    )
    
    # Get workspace
    workspace = get_workspace()
    
    for m in batch_sizes:
        m_aligned = align_to(m, 16)
        
        A_bf16 = torch.randn(m_aligned, K_aligned, device=device, dtype=ACCUM_DTYPE).contiguous()
        
        def run_once():
            # Quantize activation on-the-fly
            A_fp4 = act_quantizer.quantize_impl(A_bf16)
            # Update usage flags
            A_fp4.update_usage(rowwise_usage=True)
            W_fp4.update_usage(columnwise_usage=True)
            # FP4 GEMM via general_gemm
            # general_gemm(A, B, workspace) computes A @ B^T
            out, *_ = general_gemm(
                W_fp4,      # Weight [N, K]
                A_fp4,      # Input [M, K]  
                workspace,
                out_dtype=ACCUM_DTYPE,
            )
            return out
        
        # Warm-up
        for _ in range(10):
            run_once()
        
        torch.cuda.synchronize()
        times = []
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(num_graph_repeats):
                run_once()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        results.append(np.mean(times))
    
    return results


@torch.inference_mode()
def benchmark_fp4_pure_gemm(N, K):
    """
    Pure FP4 GEMM - both inputs pre-quantized.
    This measures only tensor core performance (upper bound).
    """
    if not HAS_FP4:
        return [0.0] * len(batch_sizes)
    
    results = []
    
    K_aligned = align_to(K, 16)
    N_aligned = align_to(N, 16)
    
    # Weight quantizer
    weight_quantizer = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_rht=False,
        with_2d_quantization=True,
        stochastic_rounding=False,
    )
    
    W_bf16 = torch.randn(N_aligned, K_aligned, device=device, dtype=ACCUM_DTYPE).contiguous()
    W_fp4 = weight_quantizer.quantize_impl(W_bf16)
    
    # Activation quantizer (no RHT for purest benchmark)
    act_quantizer = NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=False,
        with_rht=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
    )
    
    workspace = get_workspace()
    
    for m in batch_sizes:
        m_aligned = align_to(m, 16)
        
        # Pre-quantize activation (not realistic but measures pure GEMM)
        A_bf16 = torch.randn(m_aligned, K_aligned, device=device, dtype=ACCUM_DTYPE).contiguous()
        A_fp4 = act_quantizer.quantize_impl(A_bf16)
        
        def run_gemm_only():
            A_fp4.update_usage(rowwise_usage=True)
            W_fp4.update_usage(columnwise_usage=True)
            out, *_ = general_gemm(
                W_fp4,
                A_fp4,
                workspace,
                out_dtype=ACCUM_DTYPE,
            )
            return out
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.graph(graph,stream=stream):
            for _ in range(num_graph_repeats):
                run_gemm_only()
        # Warm-up
        for _ in range(num_warmups):
            graph.replay()
        
        torch.cuda.synchronize()
        times = []
        for _ in range(num_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        results.append(np.mean(times))
    
    return results


# ============ Main Benchmark ============

configs = [
    {"name": "Small", "K": 2048, "N": 1536},
    {"name": "Large", "K": 4096, "N": 3072},
]

print("\n" + "="*60)
print("Pure GEMM Benchmark: BF16 vs FP4 (using general_gemm)")
print("="*60)

all_results = {}

for config in configs:
    K = config["K"]
    N = config["N"]
    name = config["name"]
    
    print(f"\n{name} config: K={K}, N={N}")
    print("-" * 40)
    
    print("  Running BF16 matmul...")
    bf16_times = benchmark_bf16_matmul(N, K)
    
    if HAS_FP4:
        print("  Running FP4 GEMM (with activation quantization)...")
        # fp4_with_quant_times = benchmark_fp4_general_gemm(N, K)
        fp4_with_quant_times = [0.0] * len(batch_sizes)
        
        print("  Running FP4 GEMM (pure, no activation quantization)...")
        fp4_pure_times = benchmark_fp4_pure_gemm(N, K)
    else:
        fp4_with_quant_times = [0.0] * len(batch_sizes)
        fp4_pure_times = [0.0] * len(batch_sizes)
    
    all_results[name] = {
        'bf16': bf16_times,
        'fp4_quant': fp4_with_quant_times,
        'fp4_pure': fp4_pure_times,
    }
    
    print(f"\n  {'Batch':<8} {'BF16 (ms)':<12} {'FP4+quant (ms)':<15} {'FP4 pure (ms)':<15} {'Speedup':<10}")
    print("  " + "-" * 60)
    
    for i, m in enumerate(batch_sizes):
        m_aligned = align_to(m, 16)
        bf16_t = bf16_times[i]
        fp4_q_t = fp4_with_quant_times[i]
        fp4_p_t = fp4_pure_times[i]
        
        speedup = bf16_t / fp4_p_t if fp4_p_t > 0 else 0
        
        print(f"  {m_aligned:<8} {bf16_t:<12.3f} {fp4_q_t:<15.3f} {fp4_p_t:<15.3f} {speedup:<10.2f}x")

# ============ Plotting ============

fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 5))
if len(configs) == 1:
    axes = [axes]

for idx, config in enumerate(configs):
    name = config["name"]
    results = all_results[name]
    
    ax = axes[idx]
    x = np.arange(len(batch_sizes))
    width = 0.25
    
    ax.bar(x - width, results['bf16'], width, label='BF16', color='blue', alpha=0.7)
    if HAS_FP4:
        ax.bar(x, results['fp4_quant'], width, label='FP4 (+ act quant)', color='orange', alpha=0.7)
        ax.bar(x + width, results['fp4_pure'], width, label='FP4 (pure GEMM)', color='green', alpha=0.7)
    
    ax.set_xlabel('Batch Size (M)')
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'{name}: K={config["K"]}, N={config["N"]}')
    ax.set_xticks(x)
    ax.set_xticklabels([align_to(m, 16) for m in batch_sizes], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fp4_general_gemm_benchmark.png', dpi=300)
print(f"\nSaved plot to fp4_general_gemm_benchmark.png")

# ============ TFLOPS calculation ============

print("\n" + "="*60)
print("TFLOPS Analysis")
print("="*60)

for config in configs:
    K = config["K"]
    N = config["N"]
    name = config["name"]
    results = all_results[name]
    
    print(f"\n{name} config:")
    print(f"  {'Batch':<8} {'BF16 TFLOPS':<15} {'FP4 TFLOPS':<15}")
    print("  " + "-" * 40)
    
    for i, m in enumerate(batch_sizes):
        m_aligned = align_to(m, 16)
        flops = 2.0 * m_aligned * N * K
        
        bf16_tflops = (flops * num_graph_repeats) / (results['bf16'][i] * 1e-3) / 1e12
        fp4_tflops = (flops * num_graph_repeats) / (results['fp4_pure'][i] * 1e-3) / 1e12 if results['fp4_pure'][i] > 0 else 0
        
        print(f"  {m_aligned:<8} {bf16_tflops:<15.2f} {fp4_tflops:<15.2f}")