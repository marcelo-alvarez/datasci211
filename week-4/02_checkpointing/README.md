# 02_checkpointing

Checkpoint and resume workflow with SLURM signal handling. See the [Week 4 overview](../README.md) for environment instructions.

## Files
- `checkpoint_io.py` — `CheckpointManager` with atomic save/load and pruning.
- `signal_handler.py` — `SlurmSignalMonitor` for `SIGUSR1`/`SIGTERM`.
- `train_with_checkpoint.py` — single-GPU training entry point using the helpers.
- `checkpoint_train.sh` — SBATCH launcher that activates the environment and wires up signal trapping.
- `notes.md` — walkthrough and troubleshooting notes.

## Recommended Submission

```bash
sbatch 02_checkpointing/checkpoint_train.sh
```

Default resources match `01_slurm_basics`: `-A marlowe-c001 -p class --qos class --gpus 1 --cpus-per-task 4 --time 00:30:00`. Ten epochs take roughly 11 minutes; reduce with `--epochs <n>` as needed.

The SBATCH wrapper purges modules, activates `ds211-week4`, and runs `train_with_checkpoint.py`.
 
### Direct `srun`

```bash
srun -A marlowe-c001 -p class --qos class --gpus 1 --ntasks 1 --cpus-per-task 4 --time 00:30:00 \
  python 02_checkpointing/train_with_checkpoint.py --epochs 2 --save-every 1
```

Use `--simulate-preemption` to trigger SIGUSR1 without relying on SLURM preemption.

Outputs:
- Metrics: `runs/<jobid>/metrics.jsonl`
- Checkpoints: `runs/<jobid>/checkpoints/`
- Logs: `training.log`
