# DataSci 211 — Course Materials

This repository contains the student materials for Stanford DataSci 211. General tips will be added to [**TIPS.md**](TIPS.md) as we go.

### [Week 1](week-1)
Cluster warm-up examples for CPU and GPU Slurm jobs. Includes [`hello_cpu.sh`](week-1/examples/example1/hello_cpu.sh) and [`hello_gpu.sh`](week-1/examples/example2/hello_gpu.sh). See [`week-1/README.md`](week-1/README.md) for details.

### [Week 2](week-2)
GPU performance analysis using roofline model and Amdahl's Law. Includes [`run_roofline.sh`](week-2/run_roofline.sh) and [`run_amdahl.sh`](week-2/run_amdahl.sh). See [`week-2/README.md`](week-2/README.md) for setup and usage.

### [Week 4](week-4)
SLURM training workflows covering single-GPU jobs, checkpoint/resume flows, and two-GPU DDP launches. Includes `single_gpu_train.sh`, `checkpoint_train.sh`, `ddp_train.sh`, shared utilities under `week-4/common/`, and the Micromamba environment spec (`environment-week4.yml`). See [`week-4/README.md`](week-4/README.md) for activation steps and commands.

### [Week 5](week-5)
Single-node PyTorch DDP strong-scaling exercise using synthetic data. See [`week-5/README.md`](week-5/README.md) for the workload description, SLURM submission scripts, and metrics aggregation workflow.

### [Week 6](week-6)
GPU profiling lecture covering when to profile, identifying bottlenecks, and using NVIDIA Nsight Systems and NVTX, and Nsight Compute. See [`week-6/DataSci211-week-6.pdf`](week-6/DataSci211-week-6.pdf).
