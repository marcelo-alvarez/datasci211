# DataSci 211 — Week 1 Cluster Warm-Up

Use these examples to practice basic Slurm submissions on Marlowe before moving on to the GPU-focused week 2 materials.

---

## Copy the starter examples
Run the following once from a Marlowe login node to copy the shared examples into your home directory:

```bash
rsync -a /scratch/c001/examples $HOME
```

After this command you will have `~/examples/example1` and `~/examples/example2` in your home directory.

### Example 1 — CPU job
```bash
cd ~/examples/example1
sbatch hello_cpu.sh
```
- `hello_cpu.sh` runs a short Monte Carlo π approximation on two CPU tasks using `srun`.
- `sample_output.txt` shows a reference Slurm log.

### Example 2 — GPU job
```bash
cd ~/examples/example2
sbatch hello_gpu.sh
```
- `hello_gpu.sh` loads the NVIDIA HPC SDK, compiles `hello_gpu.cu`, and launches it with `srun`.
- `sample_output.txt` shows the expected output, including `nvidia-smi` diagnostics and kernel timing.

The copies of these scripts are also versioned in this repository under `week-1/examples/` for reference.
