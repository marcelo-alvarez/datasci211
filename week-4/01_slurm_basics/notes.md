# Example 01: SLURM Basics

## Overview

This example introduces single-GPU training workflows on the Marlowe cluster using SLURM. It demonstrates how to submit GPU jobs, manage Python environments in SLURM scripts, and monitor training progress. The training script (`train_single.py`) delivers a complete end-to-end workflow: synthetic data generation, model training with PyTorch, and metrics logging; it also writes basic per-epoch checkpoints so students can inspect artifacts ahead of the dedicated checkpoint/resume flow in Example 02.

This example provides the foundation for later topics:
- **Example 02 (Checkpointing)**: Extends this workflow with signal handling and graceful job termination
- **Example 03 (Distributed Training)**: Scales this workflow to multi-GPU DistributedDataParallel (DDP)

## Learning Objectives

After working through this example, instructors will be able to:
1. **Submit GPU jobs via SLURM**: Explain SBATCH directives for GPU allocation, resource requests, and output management
2. **Manage environments in batch scripts**: Demonstrate correct activation of micromamba environments within SLURM jobs
3. **Monitor training logs**: Highlight the use of `squeue`, `tail`, and SLURM output files to track job progress
4. **Interpret training metrics**: Summarize training/validation metrics from logs and JSON output files
5. **Debug common SLURM issues**: Diagnose environment conflicts, missing modules, and path problems

## Exercise Instructions

### Prerequisites

Set up the Week 4 environment once:
```bash
cd /users/marceloa/work/datasci211/datasci211-course-materials/week-4
bash setup_week4_env.sh
```

### Step 1: Local Test Run (CPU)

Before submission to SLURM, the training script should be verified locally with a small dataset:

```bash
# Activate environment
export MAMBA_ROOT_PREFIX="/scratch/c001/sw/micromamba"
eval "$($MAMBA_ROOT_PREFIX/bin/micromamba shell hook --shell=bash)"
micromamba activate ds211-week4

# Run quick test (1 epoch, tiny dataset, CPU-only)
python train_single.py \
    --n-train 32 \
    --n-val 16 \
    --n-test 16 \
    --epochs 1 \
    --num-workers 0 \
    --output-dir ./test_run
```

Expected console output:
- Console logs showing training progress
- `test_run/` directory with `training.log`, `metrics.json`, and checkpoint files

### Step 2: Submit SLURM Job (GPU)

Submit the full training job to run on a GPU:

```bash
# Create logs directory (required for SBATCH output)
mkdir -p logs

# Submit job
sbatch single_gpu_train.sh
```

Typical scheduler response:
```
Submitted batch job 123456
```

### Step 3: Monitor Job Progress

Job status can be checked with:
```bash
# View queue
squeue -u $USER

# Tail output log (replace 123456 with the job ID)
tail -f logs/single_gpu_123456.out

# Check error log (should be empty if the job succeeds)
tail logs/single_gpu_123456.err
```

Job states:
- `PD` (pending): Waiting for GPU allocation
- `R` (running): Job is executing
- `CG` (completing): Job is finishing up
- No output from `squeue`: Job completed (check logs for success/failure)

### Step 4: Examine Results

After the job completes, inspect the outputs:

```bash
# View metrics summary (replace 123456 with the job ID)
cat runs/123456/metrics.json

# Check final test accuracy
jq '.test' runs/123456/metrics.json

# List saved checkpoints
ls -lh runs/123456/checkpoint_*.pt
```

Resulting artifacts:
- `metrics.json`: Per-epoch train/val metrics and final test results
- `checkpoint_epoch_*.pt`: Model checkpoints for each epoch
- `checkpoint_best.pt`: Best model (lowest validation loss)
- `training.log`: Detailed training logs

### Step 5: Customize Training (Optional)

Hyperparameters can be adjusted by editing `single_gpu_train.sh`. The bundled defaults train for 16 epochs on ~200k synthetic examples (with ~48k each for validation and test) using a 256-wide hidden layer, which yields roughly a 2 minute run. For faster experiments, shrink the dataset and epoch count in the launcher:

```bash
# Example: quick smoke test (~30 seconds)
python train_single.py \
    --epochs 6 \
    --hidden-dim 128 \
    --n-train 20000 \
    --n-val 5000 \
    --n-test 5000 \
    --output-dir "$RUN_DIR"
```

Resubmit with `sbatch single_gpu_train.sh` after making changes.

## Common Issues

### Issue 1: Wrong Environment Active

**Symptom**: `ModuleNotFoundError: No module named 'torch'`

**Cause**: Using `ds211-python` (weeks 1-2) instead of `ds211-week4`

**Fix**: Verify the environment in the SBATCH script:
```bash
# Check which environment is active
echo "Environment: $(which python)"
micromamba list | grep pytorch  # Should show PyTorch 2.3
```

### Issue 2: Module Conflicts

**Symptom**: CUDA version mismatch errors or `libcuda.so` not found

**Cause**: Marlowe's default modules conflict with micromamba's CUDA libraries

**Fix**: Ensure `module purge` is called before activating micromamba (already in `single_gpu_train.sh`)

### Issue 3: Missing Logs Directory

**Symptom**: SLURM job fails immediately with "No such file or directory: logs/single_gpu_*.out"

**Cause**: SBATCH output directory doesn't exist

**Fix**: Create the directory before submitting:
```bash
mkdir -p logs
```

### Issue 4: Forgot `--gpus` Flag

**Symptom**: Job runs on CPU despite being submitted to GPU partition, or very slow training

**Cause**: Missing `#SBATCH --gpus=1` directive

**Fix**: Verify the SBATCH script includes GPU allocation (already present in `single_gpu_train.sh`). GPU assignment can be confirmed in the logs:
```bash
grep "GPU:" logs/single_gpu_*.out
```

### Issue 5: Permission/Path Issues

**Symptom**: `PermissionError` or "cannot create directory"

**Cause**: Running from a directory without write permissions

**Fix**: Ensure the working directory has write permissions:
```bash
cd /users/marceloa/work/datasci211/datasci211-course-materials/week-4/01_slurm_basics
```

### Issue 6: Job Gets Stuck in Queue

**Symptom**: Job stays in `PD` state for a long time

**Cause**: High cluster load or resource limits

**Fix**:
- Check queue: `squeue -p class` (see how many jobs are ahead)
- Reduce time limit if urgent: Change `#SBATCH -t 00:30:00` to `00:10:00`
- Use `scontrol show job <jobid>` to see why job is pending (look for "Reason" field)

### Issue 7: NaN Loss During Training

**Symptom**: "Training diverged: NaN loss detected!" in logs

**Cause**: Learning rate too high or numerical instability

**Fix**: Reduce learning rate:
```bash
python train_single.py --lr 0.0001  # Lower LR
```

### Debugging Tips

1. Local smoke tests (`--n-train 32 --epochs 1`) catch configuration issues before submitting to SLURM.
2. Reviewing both stdout and stderr logs ensures important errors are not missed.
3. Adding `echo "Python: $(which python)"` to SBATCH scripts confirms that the intended environment is active.
4. Absolute paths reduce ambiguity in long-running jobs and make logs easier to interpret.
5. The SLURM manual (`man sbatch` or https://slurm.schedmd.com/sbatch.html) remains the canonical reference for directive syntax.
