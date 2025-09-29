import os, random
rank = int(os.environ.get("SLURM_PROCID", 0))
random.seed(1234 + rank)

N = 200_000  # ~<1s per task on typical CPUs
inside = 0
for _ in range(N):
    x = random.random(); y = random.random()
    inside += (x*x + y*y) < 1.0

pi = 4.0 * inside / N
print(f"[task {rank}] pi≈{pi:.6f} from {N} samples on host={os.uname().nodename}")

