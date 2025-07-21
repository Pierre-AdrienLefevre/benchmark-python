# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python benchmarking tool that measures system performance across multiple dimensions:
- CPU (single-core and multi-core using joblib)
- RAM (large array operations with NumPy)
- Disk I/O (file read/write operations)
- GPU (matrix operations using PyTorch with CUDA/MPS support)
- Combined CPU+GPU workloads

The main entry point is `Benchmark.py`, which runs all benchmarks with a 10-minute timeout per test.

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
- `cpu_benchmark_singlecore()` / `cpu_benchmark_multicore()`: Fibonacci sequence computation benchmarks
- `ram_benchmark()`: Large NumPy array creation and summation
- `disk_benchmark()`: File I/O operations with random data
- `gpu_benchmark_pytorch()`: Matrix multiplication with trigonometric operations
- `combined_cpu_gpu_benchmark()`: Parallel CPU and GPU workloads using threading

The GPU benchmark automatically detects and uses available backends (CUDA for NVIDIA, MPS for Apple Silicon).

## Key Dependencies

- **NumPy**: Large array operations for memory benchmarking
- **PyTorch**: GPU computation with automatic backend detection
- **joblib**: Parallel processing for multi-core CPU benchmarks
- **threading**: Timeout handling and parallel CPU+GPU execution