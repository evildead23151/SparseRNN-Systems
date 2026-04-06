# SparseRNN-Systems 🚀  
### Block-Sparse LSTM with CUDA Acceleration for Real GPU Efficiency

> Designing sequence models that respect hardware — not just math.

---

## 🧠 Overview

WarpSparse is a GPU-aware implementation of block-sparse LSTM built to study:

- Real latency reduction (not theoretical FLOPs)
- Memory-efficient sequence modeling
- Hardware-aligned sparsity for modern GPUs

The core idea:

> Compute only what matters — and only when hardware benefits.

---

## ⚡ Key Concepts

### Memory > FLOPs
LSTM workloads are memory-bandwidth bound. Reducing memory movement matters more than reducing compute.

### Block Sparsity (Real Sparsity)
- Fixed-size tiles (e.g., 16×16)
- Entire blocks are pruned
- Enables coalesced access and warp efficiency

No element-wise masking. No wasted loads.

### Hardware-Aware Design
Built around:
- Warp execution (SIMT)
- Coalesced memory access
- Shared memory reuse
- Kernel-level optimization

---

## 🏗️ Architecture

PyTorch (baseline + training)  
↓  
C++ Extension (binding)  
↓  
CUDA Kernels  
- Dense GEMM (cuBLAS)  
- Block-Sparse MatMul  
- LSTM Pointwise Update  

---

## 📊 Current Benchmark

cuDNN LSTM:        27.216 ms  
CUDA Dense LSTM:   35.914 ms  
Sparse CUDA LSTM:  206.257 ms  

Speedup:  
Sparse vs cuDNN:      0.13×  
Sparse vs CUDA Dense: 0.17×  

---

## ⚠️ Status

Working system, not optimized yet.

### Done
- LSTM forward pass (correct)
- CUDA integration
- Block-sparse design
- Benchmark harness

### Missing
- True compressed sparse storage
- cuBLAS/CUTLASS optimization
- Kernel fusion
- Memory reuse optimization
- Backward pass

---

## 🧪 Evaluation

Metrics:
- Latency (ms)
- Throughput
- Memory usage
- Accuracy vs sparsity

---

## 🚀 Roadmap

- Block-sparse compressed storage
- cuBLASLt / tensor core usage
- Kernel fusion
- Backward pass (training)
- Nsight profiling
- Transformer comparison

---

## 🧠 Key Learnings

- GPUs hate random sparsity  
- Memory access dominates performance  
- Efficient kernels > correct math  
- “Working” ≠ “Fast”  

---

## 📌 Final Thought

The future of AI is not bigger models — it’s smarter computation.
