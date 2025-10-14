# 01_slurm_basics

Single-GPU training example with SLURM submission, logging, and synthetic data. Refer to the [Week 4 README](../README.md) for environment setup.

## Files
- `train_single.py` — command-line entry point built on the shared utilities.
- `single_gpu_train.sh` — SBATCH wrapper that activates `ds211-week4`, prepares run directories, and records metrics.
- `notes.md` — walkthrough and troubleshooting notes.

## Recommended Submission

```bash
sbatch 01_slurm_basics/single_gpu_train.sh
```

Default resources: `-A marlowe-c001 -p class --qos class --gpus 1 --cpus-per-task 4 --time 00:30:00`. The bundled script runs 16 epochs over ~200k synthetic samples (~48k val/test) with a 256-wide hidden layer, completing in about 2 minutes on a single H100. Edit the CLI flags in the script to shorten or lengthen the demo.

The SBATCH script purges modules, activates micromamba, and then launches `train_single.py`. A warning about `module 's' cannot be unloaded` is expected when SLURM’s shim module is absent.

### Direct `srun`
Useful for live demos after acquiring an interactive allocation:

```bash
srun -A marlowe-c001 -p class --qos class --gpus 1 --ntasks 1 --cpus-per-task 4 --time 00:15:00 \
  python 01_slurm_basics/train_single.py --epochs 1 --n-train 256 --n-val 64 --n-test 64
```

Logs land under `runs/<jobid>/`; metrics are written to `metrics.json`.
