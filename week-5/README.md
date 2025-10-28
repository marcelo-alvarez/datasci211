# Week 5 – DDP Strong Scaling

Week 5 evaluates single-node PyTorch DDP scaling using a synthetic ConvNet workload. The `0_gpu_ddp_strong_scaling/` module includes:

- `python/ddp_synthetic.py` – launches a configurable synthetic training run and prints throughput results (`RESULT ... samples_per_s=...`).
- `python/aggregate_scaling.py` – parses the log files from multiple runs to compute speedup/efficiency and plot `scaling.png`.
- `slurm/ddp_single_node.sbatch` plus `slurm/load_env.sh` – submit-ready SLURM scripts; sweep GPU counts with the commands listed in `slurm/README_sweep.txt`.
