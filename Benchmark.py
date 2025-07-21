import numpy as np
import os
import torch
from threading import Thread
from joblib import Parallel, delayed
import time


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

def cpu_task(iterations, loops=1):
    """
    Tâche CPU intensive : effectue une série de calculs.
    """
    for _ in range(loops):
        x, y = 1, 1
        for _ in range(iterations):
            x, y = y, x + y
    return x + y

def cpu_benchmark_singlecore(total_iterations=10_000_000, loops=5):
    """
    Benchmark CPU utilisant un seul cœur (séquentiel).
    - total_iterations : Nombre total d'itérations à exécuter.
    - loops : Nombre de boucles répétées pour répartir les calculs.
    """
    print("Starting single-core CPU benchmark...")
    iterations_per_loop = total_iterations // loops
    print(f"Each loop will handle {iterations_per_loop} iterations, repeated {loops} times.")

    start = time.time()
    for loop in range(loops):
        cpu_task(iterations_per_loop)  # Exécute la tâche séquentiellement
        print(f"Loop {loop + 1}/{loops} completed.")
    end = time.time()

    print(f"Single-core CPU benchmark completed in {end - start:.2f} seconds")
    return end - start

def cpu_benchmark_multicore(total_iterations=100_000_000, n_jobs=None, loops=50):
    """
    Benchmark multicore avec joblib utilisant tous les cœurs disponibles.
    - total_iterations : Nombre total d'itérations à exécuter.
    - n_jobs : Nombre de cœurs utilisés (None = tous les cœurs).
    - loops : Nombre de boucles répétées pour prolonger le test.
    """
    if n_jobs is None:
        n_jobs = os.cpu_count()
    print(f"Starting joblib CPU benchmark with {n_jobs} jobs...")
    iterations_per_job = total_iterations // n_jobs
    iterations_per_loop = iterations_per_job // loops
    print(f"Each job will handle {iterations_per_loop} iterations per loop, "
          f"repeated {loops} times.")

    start = time.time()
    for loop in range(loops):
        Parallel(n_jobs=n_jobs)(
            delayed(cpu_task)(iterations_per_loop) for _ in range(n_jobs)
        )
        print(f"Loop {loop + 1}/{loops} completed.")
    end = time.time()

    print(f"Joblib CPU benchmark completed in {end - start:.2f} seconds")
    return end - start

def ram_benchmark(size=2_000_000_000):  # Double la taille à 1 milliard
    print("Starting RAM benchmark...")
    start = time.time()
    data = np.random.rand(size)
    sum_data = np.sum(data)
    end = time.time()
    print(f"RAM benchmark completed in {end - start:.2f} seconds (Sum: {sum_data})")
    return end - start

def disk_benchmark(file_size=4_000_000_000):  # Fichier de 2GB
    print("Starting Disk benchmark...")
    start = time.time()
    with open("benchmark_test_file", "wb") as f:
        f.write(os.urandom(file_size))
    write_time = time.time()
    with open("benchmark_test_file", "rb") as f:
        _ = f.read()
    read_time = time.time()
    os.remove("benchmark_test_file")
    print(f"Disk Write: {write_time - start:.2f}s, Read: {read_time - write_time:.2f}s")
    return write_time - start, read_time - write_time

def gpu_benchmark_pytorch(size=15_000, loops=200):  # Matrices 5,000 x 5,000, 200 répétitions
    print("Starting extended GPU benchmark with PyTorch...")
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        device = "cuda" if torch.cuda.is_available() else "mps"  # 'mps' for Metal on macOS
        print(f"Using GPU backend: {device}")

        # Crée deux matrices sur le GPU
        start = time.time()
        matrix_a = torch.rand((size, size), device=device)
        matrix_b = torch.rand((size, size), device=device)

        # Effectue plusieurs multiplications de matrices + fonctions mathématiques
        for i in range(loops):
            result = torch.matmul(matrix_a, matrix_b)
            result = torch.sin(result)  # Applique une fonction mathématique
            if i % (loops // 10) == 0:  # Affiche la progression
                print(f"Progress: {i / loops * 100:.1f}%")

        end = time.time()
        print(f"GPU benchmark completed in {end - start:.2f} seconds")
        return end - start
    else:
        print("No GPU available for PyTorch.")
        return None


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