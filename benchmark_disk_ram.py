import numpy as np
import time
import os

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