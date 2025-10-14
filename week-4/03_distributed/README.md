# 03_distributed

PyTorch Distributed Data Parallel (DDP) example for multi-GPU runs on Marlowe. Review the [Week 4 overview](../README.md) before using these launchers.

## Files
- `ddp_utils.py` — process-group helpers, logging controls, and rank utilities.
- `train_ddp.py` — training driver with checkpoint/resume and signal handling.
- `ddp_train.sh` — two-GPU launcher that detects existing checkpoints and resumes automatically.
- `notes.md` — walkthrough and troubleshooting notes.

## Recommended Submission

```bash
sbatch 03_distributed/ddp_train.sh
```

Default resources: `-A marlowe-c001 -p class --qos class --nodes 1 --ntasks 2 --gpus 2 --gpus-per-task 1 --cpus-per-task 4 --time 01:00:00`. Five epochs take about 9 minutes. `module purge` prints the usual `module 's' cannot be unloaded` warning.

To resume, export the checkpoint directory (created by the previous run) and relaunch the same script. The training entrypoint automatically loads the most recent checkpoint when present:

```bash
CHECKPOINT_DIR=$(pwd)/runs/ddp_job_<jobid>/checkpoints \
  sbatch 03_distributed/ddp_train.sh
```

Alternatively, pass overrides through to `train_ddp.py` directly:

```bash
sbatch 03_distributed/ddp_train.sh -- --checkpoint-dir /path/to/checkpoints --output-dir /path/to/run --resume
```

### Interactive Demo

```bash
srun -A marlowe-c001 -p class --qos class --nodes 1 --ntasks 2 --gpus 2 --cpus-per-task 4 --time 00:20:00 \
  python 03_distributed/train_ddp.py --epochs 1 --global-batch-size 64 --micro-batch-size 32
```

Use `--backend gloo` for CPU-only runs.

Outputs:
- Run directory: `runs/ddp_job_<jobid>/` (or the path you provide)
- Checkpoints: `checkpoints/`
- Metrics: `metrics.jsonl`
