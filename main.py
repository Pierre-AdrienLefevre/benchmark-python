import json
import os
import platform
import socket
import time
from datetime import datetime
from threading import Thread

import psutil
import torch

from benchmark_cpu import cpu_benchmark_singlecore, cpu_benchmark_multicore
from benchmark_disk_ram import ram_benchmark, disk_benchmark
from benchmark_gpu import gpu_benchmark_pytorch


def get_system_info():
    """
    Collecte les informations détaillées du système.
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "max_frequency": psutil.cpu_freq().max if psutil.cpu_freq() else "N/A",
            "current_frequency": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A",
        },
        "memory": {
            "total": round(psutil.virtual_memory().total / (1024**3), 2),  # GB
            "available": round(psutil.virtual_memory().available / (1024**3), 2),  # GB
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        }
    }
    
    # Informations GPU détaillées
    if torch.cuda.is_available():
        info["gpu"] = {
            "type": "CUDA",
            "name": torch.cuda.get_device_name(),
            "memory_total": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),  # GB
            "compute_capability": f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}",
        }
    elif torch.backends.mps.is_available():
        info["gpu"] = {
            "type": "MPS",
            "name": "Apple Metal",
        }
    else:
        info["gpu"] = {
            "type": "None",
            "name": "No GPU acceleration available"
        }
    
    return info


def log_benchmark_result(test_name, duration, system_info,
                         log_file=f"benchmark_results-{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.log"):
    """
    Enregistre les résultats de benchmark dans un fichier log.
    """
    log_entry = {
        "test_name": test_name,
        "duration_seconds": round(duration, 2),
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info
    }
    
    # Ajouter au fichier log (format JSON Lines)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    print(f"✅ {test_name}: {duration:.2f}s - Logged to {log_file}")

def combined_cpu_gpu_benchmark(cpu_iterations=8_000_000_000, gpu_size=15_000, gpu_loops=400, n_jobs=None, loops=50):
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

    duration = end - start
    print(f"Combined CPU + GPU benchmark completed in {duration:.2f} seconds")
    return duration

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
    
    # Collecte des informations système
    system_info = get_system_info()
    
    # Liste des benchmarks à exécuter
    benchmarks = [
        ("CPU Single Core", cpu_benchmark_singlecore),
        ("CPU Multi Core", cpu_benchmark_multicore),
        ("RAM", ram_benchmark),
        ("Disk", disk_benchmark),
        ("GPU", gpu_benchmark_pytorch),
        ("Combined CPU + GPU", combined_cpu_gpu_benchmark),
    ]
    
    print("\n🚀 Starting benchmarks (no timeout)...")
    results = {}
    
    for test_name, test_func in benchmarks:
        print(f"\n--- Running {test_name} ---")
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            if result is not None:
                results[test_name] = duration
                log_benchmark_result(test_name, duration, system_info)
            else:
                print(f"❌ {test_name}: Failed (no GPU available)")
                
        except Exception as e:
            print(f"❌ {test_name}: Error - {str(e)}")
    
    print("\n📊 Final Results Summary:")
    for test_name, duration in results.items():
        print(f"  {test_name}: {duration:.2f}s")

    print(f"\n📝 Detailed results logged to: benchmark_results-{datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.log")
    print(f"🖥️  Machine: {system_info['hostname']} ({system_info['platform']['system']} {system_info['platform']['machine']})")
    print(f"⏰ Session completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")