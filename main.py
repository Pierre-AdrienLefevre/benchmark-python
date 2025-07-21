import os
import torch
from threading import Thread
import time
from benchmark_cpu import cpu_benchmark_singlecore, cpu_benchmark_multicore
from benchmark_gpu import gpu_benchmark_pytorch
from benchmark_disk_ram import ram_benchmark, disk_benchmark

def run_with_timeout(func, timeout, *args, **kwargs):
    """
    Runs a function with a timeout.
    If the function does not complete within the timeout, it is stopped.
    """
    class FunctionThread(Thread):
        def __init__(self, func, *args, **kwargs):
            super().__init__()
            self.func = func
            self.args = args
            self.kwargs = kwargs
            self.result = None

        def run(self):
            self.result = self.func(*self.args, **self.kwargs)

    thread = FunctionThread(func, *args, **kwargs)
    thread.start()
    thread.join(timeout)  # Wait for the thread to complete, up to 'timeout' seconds.

    if thread.is_alive():
        print(f"Timeout reached for {func.__name__}. Stopping the benchmark.")
        return None  # Timeout occurred
    return thread.result  # Return the function's result if completed

def combined_cpu_gpu_benchmark(cpu_iterations=50_000_000, gpu_size=10_000, gpu_loops=400, n_jobs=None, loops=50):
    """
    Benchmark combiné CPU + GPU.
    - cpu_iterations : Nombre total d'itérations pour le CPU.
    - gpu_size : Taille des matrices pour le GPU.
    - gpu_loops : Nombre de répétitions pour le GPU.
    - n_jobs : Nombre de cœurs utilisés pour le CPU (None = tous les cœurs).
    - loops : Nombre de boucles pour prolonger le test.
    """
    print("Starting combined CPU + GPU benchmark...")

    # CPU benchmark (multicore)
    def cpu_benchmark():
        print("Running CPU tasks...")
        cpu_benchmark_multicore(cpu_iterations, n_jobs=n_jobs, loops=loops)

    # GPU benchmark
    def gpu_benchmark():
        print("Running GPU tasks...")
        gpu_benchmark_pytorch(gpu_size, gpu_loops)

    # Start CPU and GPU benchmarks in parallel
    cpu_thread = Thread(target=cpu_benchmark)
    gpu_thread = Thread(target=gpu_benchmark)

    start = time.time()
    cpu_thread.start()
    gpu_thread.start()

    # Wait for both to finish
    cpu_thread.join()
    gpu_thread.join()
    end = time.time()

    print(f"Combined CPU + GPU benchmark completed in {end - start:.2f} seconds")
    return end - start

if __name__ == "__main__":
    # Display system information
    print("=== System Information ===")
    print(f"CPU cores available: {os.cpu_count()}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        print("MPS (Apple Metal) available")
    else:
        print("No GPU acceleration available")
    print("=" * 30)
    
    timeout = 2000  # Timeout in seconds
    results = {
        "CPU Single Core": run_with_timeout(cpu_benchmark_singlecore, timeout),
        "CPU Multi Core": run_with_timeout(cpu_benchmark_multicore, timeout),
        "RAM": run_with_timeout(ram_benchmark, timeout),
        "Disk": run_with_timeout(disk_benchmark, timeout),
        "GPU": run_with_timeout(gpu_benchmark_pytorch, timeout),
        "Combined CPU + GPU": run_with_timeout(combined_cpu_gpu_benchmark, timeout),
    }
    print("\nBenchmark Results:")
    for key, value in results.items():
        if value is not None:
            print(f"{key}: {value}")