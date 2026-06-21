# CUDA Projects

![C++](https://img.shields.io/badge/C++-00599C?style=flat-square&logo=cplusplus&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![OpenMP](https://img.shields.io/badge/OpenMP-006DB8?style=flat-square&logoColor=white)
![MPI](https://img.shields.io/badge/MPI-364d6e?style=flat-square&logoColor=white)

A collection of CUDA kernels and GPU-accelerated programs covering core parallel computing patterns — memory access, reductions, linear algebra, and performance benchmarking across GPU architectures.

## Projects

### AXPY
Implements `y = a*x + y` on both CPU (OpenMP) and GPU (CUDA). Reports Gflop/s and GB/s for each. Benchmarked on K80, P100, and V100 — output files included for comparison.

### array_reversal
Single-kernel in-place array reversal. Each thread swaps one pair of elements from opposite ends of the array.

### matrix_copy
2D CUDA kernel that copies a matrix element-by-element using a grid of 16x16 thread blocks. Illustrates mapping 2D thread indices to a flat memory layout.

### matrix_transpose
Three transpose implementations on the GPU:
- Naive: direct index swap, uncoalesced writes
- Shared memory tiled: loads a TILE_DIM x TILE_DIM tile into shared memory, then writes transposed — avoids uncoalesced global memory access

### reduction
Parallel sum reduction benchmarked across three GPU strategies vs a CPU OpenMP baseline:
- V0: atomic adds per thread
- V1: interleaved shared memory tree
- V2: sequential addressing (fewer divergent warps)

### matrix_mult
GPU matrix multiplication with a `CMatrix` abstraction. Includes strong-scaling results and a gnuplot script for plotting.

### jacobian_solver
CUDA + MPI Jacobi iterative solver. Includes Nsight Systems profiling job script, benchmark results, and a gnuplot animation of the converging solution.

## Build

Each project compiles with either `nvcc` or `nvc++`:

```bash
nvcc main.cu -o main.x
./main.x
```

On a cluster:
```bash
module load gcc/12.2.0 nvhpc/24.5
nvc++ main.cu -o main.x
```
