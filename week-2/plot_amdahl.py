"""Generate Amdahl's Law speedup plot from CSV data."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# Read theoretical speedup data
try:
    df = pd.read_csv("amdahl_speedup.csv")
    print("Theoretical Amdahl's Law data loaded")
except FileNotFoundError:
    print("Error: amdahl_speedup.csv not found")
    sys.exit(1)

# Try to read measured data (optional)
measured_df = None
fitted_p = None
try:
    measured_df = pd.read_csv("amdahl_measured.csv")
    print("Measured multi-GPU data loaded")

    # Try to read fitted parallel fraction
    try:
        with open("amdahl_fitted_p.txt", "r") as f:
            fitted_p = float(f.read().strip())
        print(f"Fitted parallel fraction: p = {fitted_p:.3f}")
    except FileNotFoundError:
        pass
except FileNotFoundError:
    print("No measured data found (run with --measure to generate)")

# Print summary
print("\n" + "="*80)
print("Amdahl's Law: Speedup vs Number of Processors")
print("="*80)

parallel_fractions = df['parallel_fraction'].unique()
print(f"Parallel fractions analyzed: {sorted(parallel_fractions)}")

for p in sorted(parallel_fractions):
    max_speedup = 1.0 / (1.0 - p)
    print(f"\np = {p:.3f} → Maximum speedup = {max_speedup:.2f}×")
    subset = df[df['parallel_fraction'] == p]
    print(subset[['num_gpus', 'speedup']].to_string(index=False))

if measured_df is not None:
    print("\n" + "="*80)
    print("Measured Multi-GPU MatMul Performance")
    print("="*80)
    print(measured_df.to_string(index=False))

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))

# Define colors for different parallel fractions
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(parallel_fractions)))

# Plot perfect linear speedup reference
max_gpus = df['num_gpus'].max()
ax.plot([1, max_gpus], [1, max_gpus], 'k--', linewidth=2,
        label='Perfect Linear Speedup', alpha=0.4, zorder=1)

# Plot speedup curves for each parallel fraction
for p, color in zip(sorted(parallel_fractions), colors):
    subset = df[df['parallel_fraction'] == p].sort_values('num_gpus')
    max_speedup = 1.0 / (1.0 - p)

    # Plot the curve
    # Format max_speedup: remove .0 for whole numbers (2.0 -> 2, 10.0 -> 10)
    max_str = f'{max_speedup:.0f}' if abs(max_speedup - round(max_speedup)) < 0.01 else f'{max_speedup:.1f}'
    ax.plot(subset['num_gpus'], subset['speedup'], 'o-',
            color=color, markersize=8, linewidth=2,
            label=f'p = {p:.3f} (max = {max_str}×)', zorder=10, alpha=0.6)

    # Draw horizontal line showing ceiling
    ax.axhline(max_speedup, color=color, linestyle=':', linewidth=1, alpha=0.3)

# Plot measured data if available
if measured_df is not None:
    ax.plot(measured_df['num_gpus'], measured_df['speedup'], 'r*',
            markersize=20, markeredgewidth=2, markeredgecolor='darkred',
            label='Measured (Multi-GPU MatMul)', zorder=20)

    # Plot fitted curve if available
    if fitted_p is not None:
        fitted_gpus = np.array([1, 2, 4, 8, 16])
        fitted_speedups = [1.0 / ((1 - fitted_p) + fitted_p / n) for n in fitted_gpus]
        ax.plot(fitted_gpus, fitted_speedups, 'r--',
                linewidth=3, alpha=0.8,
                label=f'Best Fit (p = {fitted_p:.3f})', zorder=15)

# Annotations
ax.text(0.05, 0.95,
        "Amdahl's Law:\nSpeedup = 1 / [(1-p) + p/N]\n\n"
        "p = parallel fraction\nN = number of processors\n\n"
        "Serial fraction (1-p) limits\nmaximum achievable speedup",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Formatting
ax.set_xlabel('Number of GPUs / Processors', fontsize=14, fontweight='bold')
ax.set_ylabel('Speedup Factor', fontsize=14, fontweight='bold')
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=2)
ax.grid(True, which='both', linestyle=':', alpha=0.3)
ax.legend(loc='lower right', fontsize=10, framealpha=0.95)

# Set axis limits (up to 16 GPUs)
ax.set_xlim(0.8, 18)
ax.set_ylim(0.8, 18)

# Set nice tick labels
ax.set_xticks([1, 2, 4, 8, 16])
ax.set_xticklabels([1, 2, 4, 8, 16])
ax.set_yticks([1, 2, 4, 8, 16])
ax.set_yticklabels([1, 2, 4, 8, 16])

# Save
plt.tight_layout()
plt.savefig('amdahl_plot.png', dpi=150, bbox_inches='tight')
print("\n✓ Amdahl's Law plot saved to: amdahl_plot.png")