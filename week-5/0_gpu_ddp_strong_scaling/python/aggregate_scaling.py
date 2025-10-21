#!/usr/bin/env python
import re, sys, json, math
import argparse
import matplotlib.pyplot as plt

pat = re.compile(r"RESULT world_size=(\d+).*avg_epoch_s=([\d.]+) samples_per_s=([\d.]+)")

def parse_files(paths):
    rows = []
    for p in paths:
        with open(p, 'r', errors='ignore') as f:
            for line in f:
                m = pat.search(line)
                if m:
                    ws = int(m.group(1))
                    t = float(m.group(2))
                    thr = float(m.group(3))
                    rows.append((ws, t, thr, p))
    rows.sort()
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="paths to log files from runs")
    ap.add_argument("--out", default="scaling.png")
    args = ap.parse_args()
    rows = parse_files(args.logs)
    if not rows:
        print("No RESULT lines found."); sys.exit(1)

    # Compute speedup/efficiency against world_size==1 baseline
    base = [r for r in rows if r[0]==1]
    if not base:
        print("Need a world_size=1 baseline in logs."); sys.exit(1)
    t1 = sum(r[1] for r in base)/len(base)

    ns = sorted(set(r[0] for r in rows))
    T = [sum(r[1] for r in rows if r[0]==n)/len([r for r in rows if r[0]==n]) for n in ns]
    S = [t1 / t for t in T]
    E = [s / n for s,n in zip(S, ns)]

    # Plot
    plt.figure()
    plt.plot(ns, S, marker='o', label='Speedup S(N)')
    plt.plot(ns, E, marker='s', label='Efficiency E(N)')
    plt.xlabel("GPUs (N)")
    plt.ylabel("Value")
    plt.title("Single-node DDP Scaling (Speedup & Efficiency)")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    # Print table
    print("N\tT(N)\tS(N)\tE(N)")
    for n,t,s,e in zip(ns,T,S,E):
        print(f"{n}\t{t:.3f}\t{s:.2f}\t{e:.2f}")

if __name__ == "__main__":
    main()
