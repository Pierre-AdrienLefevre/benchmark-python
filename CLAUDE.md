# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python benchmarking tool that measures system performance across multiple dimensions:
- CPU (single-core and multi-core using joblib with intensive mathematical operations)
- RAM (large array operations with NumPy)
- Disk I/O (file read/write operations)
- GPU (matrix operations using PyTorch with CUDA/MPS support)
- Combined CPU+GPU workloads

The main entry point is `Benchmark.py`, which runs all benchmarks with extended timeouts (33+ minutes total).

## Development Commands

This project uses `uv` as the package manager. Common commands:

```bash
# Install dependencies
uv sync

# Run the benchmark suite
python Benchmark.py

# Install new packages
uv add <package_name>
```

## Architecture

The codebase consists of a single module with specialized benchmark functions:

- `run_with_timeout()`: Thread-based timeout wrapper for all benchmarks
- `cpu_task()`: Intensive mathematical operations including trigonometry, matrix multiplication, prime number calculation, and NumPy array operations
- `cpu_benchmark_singlecore()`: Sequential CPU benchmark (500M iterations, ~5-6 minutes)
- `cpu_benchmark_multicore()`: Parallel CPU benchmark using all available cores (8B iterations, 100 loops, ~6-7 minutes)
- `ram_benchmark()`: Large NumPy array creation and summation (2B elements)
- `disk_benchmark()`: File I/O operations with 4GB random data
- `gpu_benchmark_pytorch()`: Matrix multiplication with trigonometric operations (15k×15k matrices)
- `combined_cpu_gpu_benchmark()`: Parallel CPU and GPU workloads using threading

The CPU benchmarks automatically detect and utilize all available CPU cores. The GPU benchmark automatically detects and uses available backends (CUDA for NVIDIA, MPS for Apple Silicon).

## Key Dependencies

- **NumPy**: Large array operations for memory benchmarking and CPU mathematical operations
- **PyTorch**: GPU computation with automatic backend detection
- **joblib**: Parallel processing for multi-core CPU benchmarks
- **threading**: Timeout handling and parallel CPU+GPU execution