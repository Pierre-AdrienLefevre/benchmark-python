import time
import os
import numpy as np
from joblib import Parallel, delayed

def cpu_task(iterations, loops=1):
    """
    Tâche CPU intensive : calculs mathématiques complexes combinés.
    Inclut multiplication matricielle, trigonométrie, et opérations vectorielles.
    """
    import math
    total_result = 0

    for loop in range(loops):
        # 1. Calculs trigonométriques intensifs
        trig_sum = 0
        for i in range(iterations // 4):
            x = i * 0.001
            trig_sum += math.sin(x) * math.cos(x) + math.tan(x / (1 + x))

        # 2. Multiplications matricielles avec NumPy (plus intensives)
        matrix_result = 0
        matrix_size = min(800, int(math.sqrt(iterations // 4)))
        if matrix_size > 10:
            # Plusieurs multiplications pour être plus intensif
            for _ in range(3):
                a = np.random.rand(matrix_size, matrix_size)
                b = np.random.rand(matrix_size, matrix_size)
                c = np.dot(a, b)
                matrix_result += np.sum(c * c)  # Opération supplémentaire

        # 3. Calculs de nombres premiers (beaucoup plus intensif)
        prime_count = 0
        max_n = min(50000, iterations // 5)  # Augmenté significativement
        for n in range(2, max_n):
            is_prime = True
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    is_prime = False
                    break
            if is_prime:
                prime_count += 1

        # 4. Opérations sur tableaux NumPy (beaucoup plus intensives)
        if iterations > 1000:
            array_size = min(500000, iterations)
            # Plusieurs opérations complexes
            arr1 = np.random.rand(array_size)
            arr2 = np.random.rand(array_size)
            # Calculs très gourmands
            array_result = np.sum(np.sqrt(arr1) * np.log1p(arr1) * np.sin(arr2))
            array_result += np.sum(np.exp(arr1 * 0.01) * np.cos(arr2))
        else:
            array_result = 0

        total_result += trig_sum + matrix_result + prime_count + array_result

    return total_result

def cpu_benchmark_singlecore(total_iterations=500_000_000, loops=5):
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

def cpu_benchmark_multicore(total_iterations=8_000_000_000, n_jobs=None, loops=100):
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