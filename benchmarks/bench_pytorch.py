"""
bench_pytorch.py
================
Benchmarks PyTorch (CPU) for the same operations as bench_softcuda.cpp so we
can make an apples-to-apples CPU comparison.

Operations benchmarked:
  1. Element-wise ADD  (1M float32)
  2. Matmul 512×512
  3. XOR training (10 000 epochs, 2-layer MLP, SGD, CPU)

Run:
    cd /home/wslarch/Documents/projects/soft-cuda
    source .venv/bin/activate
    python3 benchmarks/bench_pytorch.py
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)   # single-threaded for fair comp

REPS = 20

def hline():
    print("─" * 65)

def result(label, ms, gflops=None):
    if gflops:
        print(f"  {label:<34}  {ms:8.2f} ms   {gflops:6.2f} GFLOPs")
    else:
        print(f"  {label:<34}  {ms:8.2f} ms")

# Element-wise ADD
hline()
print("  Benchmark 1: Element-wise ADD  (1M float32)  [PyTorch CPU]")
hline()

N = 1024 * 1024
a = torch.ones(N)
b = torch.ones(N)

times = []
for _ in range(REPS):
    t0 = time.perf_counter()
    c = a + b
    _ = c[0].item()   # force eval
    times.append((time.perf_counter() - t0) * 1e3)

result("ADD [PyTorch CPU]", sum(times)/len(times), N * 1e-9)
print()

# Matmul 512×512
hline()
print("  Benchmark 2: Matmul 512×512  [PyTorch CPU]")
hline()

M, K, Np = 512, 512, 512
A = torch.randn(M, K)
B = torch.randn(K, Np)
total_flops = 2 * M * K * Np

times = []
for _ in range(REPS):
    t0 = time.perf_counter()
    C = torch.mm(A, B)
    _ = C[0, 0].item()
    times.append((time.perf_counter() - t0) * 1e3)

result("Matmul 512×512 [PyTorch OpenBLAS]",
       sum(times)/len(times), total_flops * 1e-9)
print()

# XOR training
hline()
print("  Benchmark 3: XOR Training  (10 000 epochs, SGD lr=0.05, CPU)")
hline()

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 4)
        self.l2 = nn.Linear(4, 1)
    def forward(self, x):
        return self.l2(torch.relu(self.l1(x)))

model = XORNet()
opt = optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

EPOCHS = 10000
t0 = time.perf_counter()
for _ in range(EPOCHS):
    opt.zero_grad()
    pred = model(X)
    loss = loss_fn(pred, Y)
    loss.backward()
    opt.step()
elapsed = (time.perf_counter() - t0) * 1e3

result(f"XOR training [{EPOCHS} epochs, PyTorch CPU]", elapsed)
print(f"  Final loss: {loss.item():.6f}")
print()

# 100-Million Element Reduction 
hline()
print("  Benchmark 5: 100-Million Element Reduction (Mean)  [PyTorch CPU]")
hline()

N_billion = 100 * 1000 * 1000
print("  [Allocating 400MB Host Memory...]")
# Use torch.ones to mimic the C++ benchmark
A_billion = torch.ones(N_billion, dtype=torch.float32)

times = []
for _ in range(5): # Faster now
    t0 = time.perf_counter()
    M = A_billion.mean()
    _ = M.item()
    times.append((time.perf_counter() - t0) * 1e3)

result("Mean 100M [PyTorch CPU]", sum(times)/len(times))
print(f"  (Mean={M.item():.1f})")
print()

hline()
print("  For GPU comparison run bench_softcuda with SC_BACKEND_GPU.")
hline()
