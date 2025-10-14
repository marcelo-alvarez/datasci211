# Common Utilities

Shared building blocks used across all Week 4 training workflows. Refer back to the top-level [Week 4 README](../README.md) and the sprint briefs under [`dev/`](../dev) for broader context.

## Contents
- `cli.py` – Shared argument parser extensions (`add_classification_args`, `resolve_output_paths`).
- `data_utils.py` – Synthetic dataset factory + `DataLoader` wrappers (with optional DDP samplers).
- `logging_utils.py` – Console/file logging setup for single-rank and multi-rank jobs.
- `metrics.py` – JSON / JSONL append helpers for per-epoch and rank-filtered metrics.
- `random_state.py` – Save/restore helpers for Python, NumPy, CPU, and CUDA RNG states.
- `simple_model.py` – Configurable MLP via `ModelConfig`/`build_model` used in all Week 4 runs.
- `synthetic_data.py` – Legacy generation + split helpers (still referenced by tests).
- `training_utils.py` – Mini training/eval loops, metric aggregation, seeding utilities, and checkpoint helpers.

## Usage Notes
- Import these modules from any script within the repo (entry points add the repo root to `sys.path`).
- Functions assume the Week 4 micromamba environment (`ds211-week4`) so that PyTorch 2.3 + CUDA 12.1 are present.
- DDP-aware utilities rely on `torch.distributed`; when running outside SLURM/torchrun, pass `--backend gloo` for CPU demos.
- Metrics helpers default to rank-0 emission; pass `master_only=False` when you need per-rank logging for debugging.
