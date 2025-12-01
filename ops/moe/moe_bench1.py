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
        # import transformer_engine.pytorch as te
        # import transformer_engine_torch as tex
        # from transformer_engine.common import recipe
        # from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer, NVFP4Tensor
        # from transformer_engine.pytorch.cpp_extensions import general_gemm
        # from transformer_engine.pytorch.module.base import get_workspace
        HAS_FP4 = True
        print("FP4 support: ENABLED (Blackwell GPU detected)")
    except (ImportError, AttributeError) as e:
        print(f"FP4 support: DISABLED ({e})")
else:
    cap = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    print(f"FP4 support: DISABLED (requires Blackwell SM100+, current: SM{cap[0]}{cap[1]})")

ACCUM_DTYPE = torch.bfloat16
batch_sizes = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072])
num_repeats = 10
num_warmups = 2
num_graph_repeats = 100


def align_to(x, alignment):
    return ((x + alignment - 1) // alignment) * alignment


@torch.inference_mode()
def benchmark_bf16_swiglu(hidden_size, intermediate_size):
    """
    BF16 SwiGLU MLP benchmark:
    - GEMM1: x @ W_gate_up -> [M, 2*intermediate]
    - SwiGLU: gate * swish(up)  
    - GEMM2: act @ W_down -> [M, hidden]
    """
    results = []
    
    H = hidden_size
    I = intermediate_size
    
    # Weights: gate+up fused [H, 2*I], down [I, H]
    W_gate_up = torch.randn(H, 2 * I, device=device, dtype=ACCUM_DTYPE).contiguous()
    W_down = torch.randn(I, H, device=device, dtype=ACCUM_DTYPE).contiguous()
    
    for m in batch_sizes:
        print(f"Benchmarking BF16 SwiGLU: M={m}, H={H}, I={I}")
        m_aligned = align_to(m, 16)
        
        x = torch.randn(m_aligned, H, device=device, dtype=ACCUM_DTYPE).contiguous()
        # Pre-allocate buffers
        up_buf = torch.empty(m_aligned, 2 * I, device=device, dtype=ACCUM_DTYPE)
        down_buf = torch.empty(m_aligned, H, device=device, dtype=ACCUM_DTYPE)
        
        def run_swiglu():
            # GEMM1: [M, H] @ [H, 2*I] -> [M, 2*I]
            torch.matmul(x, W_gate_up, out=up_buf)
            # Split into gate and up projections
            gate = up_buf[:, :I]
            up = up_buf[:, I:]
            # SwiGLU activation: gate * silu(up)
            # silu(x) = x * sigmoid(x)
            act = gate * torch.nn.functional.silu(up)
            # GEMM2: [M, I] @ [I, H] -> [M, H]
            torch.matmul(act, W_down, out=down_buf)
            return down_buf
        
        # Capture CUDA graph
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        
        # Warmup before graph capture
        for _ in range(3):
            run_swiglu()
        torch.cuda.synchronize()
        
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(num_graph_repeats):
                run_swiglu()
        
        # Warm-up graph
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


# @torch.inference_mode()
# def benchmark_fp4_swiglu_with_quant(hidden_size, intermediate_size):
#     """
#     FP4 SwiGLU MLP with on-the-fly activation quantization (realistic inference).
#     """
#     if not HAS_FP4:
#         return [0.0] * len(batch_sizes)
    
#     results = []
    
#     H = align_to(hidden_size, 16)
#     I = align_to(intermediate_size, 16)
    
#     # Weight quantizer
#     weight_quantizer = NVFP4Quantizer(
#         fp4_dtype=tex.DType.kFloat4E2M1,
#         rowwise=True,
#         columnwise=True,
#         with_rht=False,
#         with_2d_quantization=True,
#         stochastic_rounding=False,
#     )
    
#     # Pre-quantize weights
#     # W_gate_up: [2*I, H] for general_gemm (computes A @ B^T)
#     # W_down: [H, I] for general_gemm
#     W_gate_up_bf16 = torch.randn(2 * I, H, device=device, dtype=ACCUM_DTYPE).contiguous()
#     W_down_bf16 = torch.randn(H, I, device=device, dtype=ACCUM_DTYPE).contiguous()
    
#     W_gate_up_fp4 = weight_quantizer.quantize_impl(W_gate_up_bf16)
#     W_down_fp4 = weight_quantizer.quantize_impl(W_down_bf16)
    
#     # Activation quantizer with RHT
#     act_quantizer = NVFP4Quantizer(
#         fp4_dtype=tex.DType.kFloat4E2M1,
#         rowwise=True,
#         columnwise=False,
#         with_rht=True,
#         with_post_rht_amax=True,
#         with_2d_quantization=False,
#         stochastic_rounding=False,
#     )
    
#     workspace = get_workspace()
    
#     for m in batch_sizes:
#         print(f"Benchmarking FP4 SwiGLU with quant: M={m}, H={H}, I={I}")
#         m_aligned = align_to(m, 16)
        
#         x_bf16 = torch.randn(m_aligned, H, device=device, dtype=ACCUM_DTYPE).contiguous()
        
#         def run_swiglu_fp4():
#             # Quantize input activation
#             x_fp4 = act_quantizer.quantize_impl(x_bf16)
#             x_fp4.update_usage(rowwise_usage=True)
#             W_gate_up_fp4.update_usage(columnwise_usage=True)
            
#             # GEMM1: [M, H] @ [H, 2*I] -> [M, 2*I]
#             # general_gemm computes W @ x^T, so we pass (W_gate_up, x)
#             up_buf, *_ = general_gemm(
#                 W_gate_up_fp4,  # [2*I, H]
#                 x_fp4,          # [M, H]
#                 workspace,
#                 out_dtype=ACCUM_DTYPE,
#             )
            
#             # SwiGLU activation (in BF16)
#             gate = up_buf[:, :I]
#             up = up_buf[:, I:]
#             act = gate * torch.nn.functional.silu(up)
            
#             # Quantize intermediate activation for GEMM2
#             act_fp4 = act_quantizer.quantize_impl(act.contiguous())
#             act_fp4.update_usage(rowwise_usage=True)
#             W_down_fp4.update_usage(columnwise_usage=True)
            
#             # GEMM2: [M, I] @ [I, H] -> [M, H]
#             out, *_ = general_gemm(
#                 W_down_fp4,  # [H, I]
#                 act_fp4,     # [M, I]
#                 workspace,
#                 out_dtype=ACCUM_DTYPE,
#             )
#             return out
        
#         # Warm-up
#         for _ in range(10):
#             run_swiglu_fp4()
        
#         torch.cuda.synchronize()
#         times = []
#         for _ in range(num_repeats):
#             start = torch.cuda.Event(enable_timing=True)
#             end = torch.cuda.Event(enable_timing=True)
#             start.record()
#             for _ in range(num_graph_repeats):
#                 run_swiglu_fp4()
#             end.record()
#             torch.cuda.synchronize()
#             times.append(start.elapsed_time(end))
        
#         results.append(np.mean(times))
    
#     return results


# @torch.inference_mode()
# def benchmark_fp4_swiglu_pure(hidden_size, intermediate_size):
#     """
#     Pure FP4 SwiGLU MLP - pre-quantized activations (upper bound performance).
#     Note: This is NOT realistic since activations change per input,
#     but measures pure FP4 tensor core throughput.
#     """
#     if not HAS_FP4:
#         return [0.0] * len(batch_sizes)
    
#     results = []
    
#     H = align_to(hidden_size, 16)
#     I = align_to(intermediate_size, 16)
    
#     # Weight quantizer
#     weight_quantizer = NVFP4Quantizer(
#         fp4_dtype=tex.DType.kFloat4E2M1,
#         rowwise=True,
#         columnwise=True,
#         with_rht=False,
#         with_2d_quantization=True,
#         stochastic_rounding=False,
#     )
    
#     # Pre-quantize weights
#     W_gate_up_bf16 = torch.randn(2 * I, H, device=device, dtype=ACCUM_DTYPE).contiguous()
#     W_down_bf16 = torch.randn(H, I, device=device, dtype=ACCUM_DTYPE).contiguous()
    
#     W_gate_up_fp4 = weight_quantizer.quantize_impl(W_gate_up_bf16)
#     W_down_fp4 = weight_quantizer.quantize_impl(W_down_bf16)
    
#     # Activation quantizer (no RHT for purest benchmark)
#     act_quantizer = NVFP4Quantizer(
#         fp4_dtype=tex.DType.kFloat4E2M1,
#         rowwise=True,
#         columnwise=False,
#         with_rht=False,
#         with_2d_quantization=False,
#         stochastic_rounding=False,
#     )
    
#     workspace = get_workspace()
    
#     for m in batch_sizes:
#         print(f"Benchmarking FP4 pure SwiGLU: M={m}, H={H}, I={I}")
#         m_aligned = align_to(m, 16)
        
#         # Pre-quantize input (not realistic but measures pure GEMM)
#         x_bf16 = torch.randn(m_aligned, H, device=device, dtype=ACCUM_DTYPE).contiguous()
#         x_fp4 = act_quantizer.quantize_impl(x_bf16)
        
#         # Pre-quantize intermediate activation (for pure GEMM measurement)
#         act_bf16 = torch.randn(m_aligned, I, device=device, dtype=ACCUM_DTYPE).contiguous()
#         act_fp4 = act_quantizer.quantize_impl(act_bf16)
        
#         def run_swiglu_pure():
#             x_fp4.update_usage(rowwise_usage=True)
#             W_gate_up_fp4.update_usage(columnwise_usage=True)
            
#             # GEMM1
#             up_buf, *_ = general_gemm(
#                 W_gate_up_fp4,
#                 x_fp4,
#                 workspace,
#                 out_dtype=ACCUM_DTYPE,
#             )
            
#             # SwiGLU (in BF16)
#             gate = up_buf[:, :I]
#             up = up_buf[:, I:]
#             _ = gate * torch.nn.functional.silu(up)
            
#             # GEMM2 with pre-quantized activation
#             act_fp4.update_usage(rowwise_usage=True)
#             W_down_fp4.update_usage(columnwise_usage=True)
            
#             out, *_ = general_gemm(
#                 W_down_fp4,
#                 act_fp4,
#                 workspace,
#                 out_dtype=ACCUM_DTYPE,
#             )
#             return out
        
#         # Capture CUDA graph
#         graph = torch.cuda.CUDAGraph()
#         stream = torch.cuda.Stream()
        
#         # Warmup before graph capture
#         for _ in range(3):
#             run_swiglu_pure()
#         torch.cuda.synchronize()
        
#         with torch.cuda.graph(graph, stream=stream):
#             for _ in range(num_graph_repeats):
#                 run_swiglu_pure()
        
#         # Warm-up graph
#         for _ in range(num_warmups):
#             graph.replay()
        
#         torch.cuda.synchronize()
#         times = []
#         for _ in range(num_repeats):
#             start = torch.cuda.Event(enable_timing=True)
#             end = torch.cuda.Event(enable_timing=True)
#             start.record()
#             graph.replay()
#             end.record()
#             torch.cuda.synchronize()
#             times.append(start.elapsed_time(end))
        
#         results.append(np.mean(times))
    
#     return results


# ============ Main Benchmark ============

# Model configurations (matching your first file)
configs = [
    {"name": "Qwen3-30B", "hidden_size": 2048, "intermediate_size": 768},
    {"name": "Qwen3-235B", "hidden_size": 4096, "intermediate_size": 1536},
]

print("\n" + "="*60)
print("SwiGLU MLP Benchmark: BF16 vs FP4 (using general_gemm)")
print("="*60)

all_results = {}

for config in configs:
    H = config["hidden_size"]
    I = config["intermediate_size"]
    name = config["name"]
    
    print(f"\n{name}: hidden={H}, intermediate={I}")
    print("-" * 40)
    
    print("  Running BF16 SwiGLU...")
    bf16_times = benchmark_bf16_swiglu(H, I)
    
    if HAS_FP4:
        print("  Running FP4 SwiGLU (with activation quantization)...")
        # fp4_with_quant_times = benchmark_fp4_swiglu_with_quant(H, I)
        
        # print("  Running FP4 SwiGLU (pure, pre-quantized activations)...")
        # fp4_pure_times = benchmark_fp4_swiglu_pure(H, I)
        fp4_with_quant_times = [0.0] * len(batch_sizes)
        fp4_pure_times = [0.0] * len(batch_sizes)
    else:
        fp4_with_quant_times = [0.0] * len(batch_sizes)
        fp4_pure_times = [0.0] * len(batch_sizes)
    
    all_results[name] = {
        'bf16': bf16_times,
        'fp4_quant': fp4_with_quant_times,
        'fp4_pure': fp4_pure_times,
        'hidden_size': H,
        'intermediate_size': I,
    }
    
    print(f"\n  {'Batch':<8} {'BF16 (ms)':<12} {'FP4+quant (ms)':<15} {'FP4 pure (ms)':<15} {'Speedup':<10}")
    print("  " + "-" * 70)
    
    for i, m in enumerate(batch_sizes):
        m_aligned = align_to(m, 16)
        bf16_t = bf16_times[i]
        fp4_q_t = fp4_with_quant_times[i]
        fp4_p_t = fp4_pure_times[i]
        
        speedup_quant = bf16_t / fp4_q_t if fp4_q_t > 0 else 0
        speedup_pure = bf16_t / fp4_p_t if fp4_p_t > 0 else 0
        
        print(f"  {m_aligned:<8} {bf16_t:<12.3f} {fp4_q_t:<15.3f} {fp4_p_t:<15.3f} {speedup_pure:<10.2f}x")

# ============ Plotting ============

fig, axes = plt.subplots(1, len(configs), figsize=(8 * len(configs), 6))
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
    ax.set_title(f'{name}: H={results["hidden_size"]}, I={results["intermediate_size"]}')
    ax.set_xticks(x)
    ax.set_xticklabels([align_to(m, 16) for m in batch_sizes], rotation=45, fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fp4_swiglu_benchmark.png', dpi=300)
print(f"\nSaved plot to fp4_swiglu_benchmark.png")

# ============ TFLOPS calculation ============

print("\n" + "="*60)
print("TFLOPS Analysis (SwiGLU MLP)")
print("="*60)

for config in configs:
    name = config["name"]
    results = all_results[name]
    H = results["hidden_size"]
    I = results["intermediate_size"]
    
    print(f"\n{name}: H={H}, I={I}")
    print(f"  {'Batch':<8} {'BF16 TFLOPS':<15} {'FP4+quant TFLOPS':<18} {'FP4 pure TFLOPS':<15}")
    print("  " + "-" * 60)
    
    for i, m in enumerate(batch_sizes):
        m_aligned = align_to(m, 16)
        # SwiGLU FLOPs: GEMM1 (M*H*2I*2) + GEMM2 (M*I*H*2) = 2*M*H*2I + 2*M*I*H = 6*M*H*I
        flops = 6.0 * m_aligned * H * I
        
        bf16_tflops = (flops * num_graph_repeats) / (results['bf16'][i] * 1e-3) / 1e12
        fp4_quant_tflops = (flops * num_graph_repeats) / (results['fp4_quant'][i] * 1e-3) / 1e12 if results['fp4_quant'][i] > 0 else 0
        fp4_pure_tflops = (flops * num_graph_repeats) / (results['fp4_pure'][i] * 1e-3) / 1e12 if results['fp4_pure'][i] > 0 else 0
        
        print(f"  {m_aligned:<8} {bf16_tflops:<15.2f} {fp4_quant_tflops:<18.2f} {fp4_pure_tflops:<15.2f}")

# ============ FLOPs vs Time Plot ============

fig2, ax2 = plt.subplots(figsize=(10, 6))

colors = plt.get_cmap('tab10')
for idx, config in enumerate(configs):
    name = config["name"]
    results = all_results[name]
    H = results["hidden_size"]
    I = results["intermediate_size"]
    
    # FLOPs per batch (in GFLOPs)
    flops_g = 6.0 * np.array([align_to(m, 16) for m in batch_sizes]) * H * I / 1e9
    
    color = colors(idx)
    ax2.plot(flops_g, results['bf16'], marker='o', linestyle='-', color=color, 
             label=f"{name} - BF16")
    if HAS_FP4:
        ax2.plot(flops_g, results['fp4_quant'], marker='^', linestyle='--', color=color, 
                 alpha=0.7, label=f"{name} - FP4 (+ quant)")
        ax2.plot(flops_g, results['fp4_pure'], marker='s', linestyle=':', color=color, 
                 alpha=0.5, label=f"{name} - FP4 (pure)")

ax2.set_xlabel("FLOPs per batch (GFLOPs)")
ax2.set_ylabel("Avg Execution Time (ms)")
ax2.set_title("SwiGLU MLP: Time vs FLOPs (BF16 vs FP4)")
ax2.legend(ncol=2, fontsize=9)
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_xscale('log')
ax2.set_yscale('log')

fig2.tight_layout()
fig2.savefig('fp4_swiglu_flops_vs_time.png', dpi=300)
print(f"Saved FLOPs vs time plot to fp4_swiglu_flops_vs_time.png")