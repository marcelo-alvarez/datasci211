# Distributed Data Parallel Example Notes

## Learning Objectives
- Understand how PyTorch Distributed Data Parallel (DDP) builds on single-GPU training.
- Practice configuring SLURM jobs to launch multi-GPU workloads on Marlowe.
- Explore gradient accumulation to match large global batch sizes with limited per-GPU memory.
- Demonstrate safe checkpoint/resume patterns when jobs are preempted.

## Classroom Flow
1. **Recap Sprint D:** Review checkpointing utilities, signal handling, and single-GPU workflows.
2. **DDP Concepts:** Discuss world size, ranks, NCCL backend, and SLURM environment variables.
3. **Live Demo:** Launch `ddp_train.sh` (2 GPUs) and inspect per-rank logs, metrics JSONL, and checkpoints.
4. **Failure Drill:** Send a SIGUSR1 to trigger graceful checkpointing, then relaunch `ddp_train.sh` with `CHECKPOINT_DIR` pointing at the saved checkpoints to demonstrate automatic resume.
5. **Extension Ideas:** Outline multi-node scaling and debugging tips (NCCL debug flags, env overrides).

## Artifacts Produced
- `runs/ddp_job_<id>/metrics.jsonl`: epoch-wise train/val/test entries aggregated on rank 0.
- `checkpoints/ddp_epoch_<n>.pt`: rank-aware checkpoint bundles with RNG state and optimizer state.
- Per-rank logs `training_rank{rank}.log` capturing distributed init, sampler state, and progress.

## Exercises
- Modify `GLOBAL_BATCH_SIZE` and `MICRO_BATCH_SIZE` to observe gradient accumulation behaviour.
- Introduce artificial preemption (`scancel --signal=USR1`) and verify resume continuity.
- Vary NCCL settings (e.g., `NCCL_DEBUG=info`) to practise diagnosing communication issues.

## Expected Outcomes
- Students can reason about how data partitioning and synchronization affect throughput.
- Teams are comfortable editing SBATCH scripts for custom allocations (GPUs, nodes, walltime).
- Consistent metrics and checkpoint artifacts across ranks validate deterministic seeding.
