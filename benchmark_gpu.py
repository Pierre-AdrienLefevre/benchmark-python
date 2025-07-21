import torch
import time
def gpu_benchmark_pytorch(size=15_000, loops=200):  # Matrices 15,000 x 15,000, 200 répétitions
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