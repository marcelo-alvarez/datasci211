"""Generate Amdahl's Law speedup plot from CSV data.

Visualizes Amdahl's Law speedup curves for various parallel fractions and
optionally overlays measured multi-GPU performance. Shows speedup ceilings
and diminishing returns. See README.md for Amdahl's Law explanation.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (for headless environments)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# ==================================================================
# LOAD DATA
# ==================================================================

# Load theoretical Amdahl's Law curves (required)
# CSV format: parallel_fraction, num_gpus, speedup, max_speedup
try:
    df = pd.read_csv("amdahl_speedup.csv")
    print("Theoretical Amdahl's Law data loaded")
except FileNotFoundError:
    print("Error: amdahl_speedup.csv not found")
    sys.exit(1)

# Load measured multi-GPU performance data (optional)
# CSV format: num_gpus, time_ms, speedup, efficiency
measured_df = None
fitted_p = None
try:
    measured_df = pd.read_csv("amdahl_measured.csv")
    print("Measured multi-GPU data loaded")

    # Load fitted parallel fraction from curve fitting
    try:
        with open("amdahl_fitted_p.txt", "r") as f:
            fitted_p = float(f.read().strip())
        print(f"Fitted parallel fraction: p = {fitted_p:.3f}")
    except FileNotFoundError:
        pass
except FileNotFoundError:
    print("No measured data found (run with --measure to generate)")

# ==================================================================
# PRINT SUMMARY
# ==================================================================
print("\n" + "="*80)
print("Amdahl's Law: Speedup vs Number of Processors")
print("="*80)

# Display theoretical speedup for each parallel fraction
parallel_fractions = df['parallel_fraction'].unique()
print(f"Parallel fractions analyzed: {sorted(parallel_fractions)}")

for p in sorted(parallel_fractions):
    # Maximum speedup = 1 / serial_fraction
    max_speedup = 1.0 / (1.0 - p)
    print(f"\np = {p:.3f} → Maximum speedup = {max_speedup:.2f}×")
    subset = df[df['parallel_fraction'] == p]
    print(subset[['num_gpus', 'speedup']].to_string(index=False))

# Display measured performance if available
if measured_df is not None:
    print("\n" + "="*80)
    print("Measured Multi-GPU MatMul Performance")
    print("="*80)
    print(measured_df.to_string(index=False))

# ==================================================================
# CREATE AMDAHL'S LAW PLOT
# ==================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Color scheme for different parallel fractions
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(parallel_fractions)))

# REFERENCE LINE: Perfect linear speedup (ideal but impossible)
# N GPUs → N× speedup (slope = 1 on log-log plot)
max_gpus = df['num_gpus'].max()
ax.plot([1, max_gpus], [1, max_gpus], 'k--', linewidth=2,
        label='Perfect Linear Speedup', alpha=0.4, zorder=1)

# THEORETICAL CURVES: Plot Amdahl's Law for each parallel fraction
for p, color in zip(sorted(parallel_fractions), colors):
    subset = df[df['parallel_fraction'] == p].sort_values('num_gpus')
    max_speedup = 1.0 / (1.0 - p)  # Speedup ceiling

    # Format ceiling label (clean formatting for whole numbers)
    max_str = f'{max_speedup:.0f}' if abs(max_speedup - round(max_speedup)) < 0.01 else f'{max_speedup:.1f}'

    # Plot speedup curve vs GPU count
    ax.plot(subset['num_gpus'], subset['speedup'], 'o-',
            color=color, markersize=8, linewidth=2,
            label=f'p = {p:.3f} (max = {max_str}×)', zorder=10, alpha=0.6)

    # Draw horizontal line showing speedup ceiling
    # No matter how many GPUs you add, speedup can't exceed this
    ax.axhline(max_speedup, color=color, linestyle=':', linewidth=1, alpha=0.3)

# MEASURED DATA: Overlay actual multi-GPU measurements
if measured_df is not None:
    # Plot measured speedup points
    ax.plot(measured_df['num_gpus'], measured_df['speedup'], 'r*',
            markersize=20, markeredgewidth=2, markeredgecolor='darkred',
            label='Measured (Multi-GPU MatMul)', zorder=20)

    # Plot fitted theoretical curve
    # This shows which Amdahl curve best explains the measurements
    if fitted_p is not None:
        fitted_gpus = np.array([1, 2, 4, 8, 16])
        # Calculate theoretical speedup using fitted p
        fitted_speedups = [1.0 / ((1 - fitted_p) + fitted_p / n) for n in fitted_gpus]
        ax.plot(fitted_gpus, fitted_speedups, 'r--',
                linewidth=3, alpha=0.8,
                label=f'Best Fit (p = {fitted_p:.3f})', zorder=15)

# ==================================================================
# ANNOTATIONS: Explain Amdahl's Law
# ==================================================================
# Text box explaining the formula
ax.text(0.05, 0.95,
        "Amdahl's Law:\nSpeedup = 1 / [(1-p) + p/N]\n\n"
        "p = parallel fraction\nN = number of processors\n\n"
        "Serial fraction (1-p) limits\n speedup",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ==================================================================
# FORMATTING AND LAYOUT
# ==================================================================
ax.set_xlabel('Number of GPUs / Processors', fontsize=14, fontweight='bold')
ax.set_ylabel('Speedup Factor', fontsize=14, fontweight='bold')

# Use log scale (base 2) to clearly show powers-of-2 GPU counts
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)

# Grid for readability
ax.grid(True, which='both', linestyle=':', alpha=0.3)

# Legend with all curves
ax.legend(loc='lower right', fontsize=10, framealpha=0.95)

# Axis limits: 1 to 16 GPUs (typical multi-GPU node)
ax.set_xlim(0.8, 18)
ax.set_ylim(0.8, 18)

# Clean tick labels (powers of 2)
ax.set_xticks([1, 2, 4, 8, 16])
ax.set_xticklabels([1, 2, 4, 8, 16])
ax.set_yticks([1, 2, 4, 8, 16])
ax.set_yticklabels([1, 2, 4, 8, 16])

# Save plot to file
plt.tight_layout()
plt.savefig('amdahl_plot.png', dpi=150, bbox_inches='tight')
print("\n✓ Amdahl's Law plot saved to: amdahl_plot.png")