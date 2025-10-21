#!/usr/bin/env python

"""
ddp_synthetic.py — single-node multi-GPU DDP scaling demo (PyTorch)

- Synthetic image classification workload (ConvNet on random tensors) to avoid I/O effects.
- Measures throughput (samples/sec) and time/epoch across world sizes.
- Uses NCCL backend (GPU → GPU). Assumes a single node with N GPUs.

USAGE (single node):
  torchrun --nproc_per_node=4 python ddp_synthetic.py --epochs 3 --batch-size 256 --model conv --flops-scale 1.0
  # nproc_per_node should equal number of visible GPUs on the node

TIPS:
- Increase --batch-size and/or --flops-scale to ensure the GPU is compute-bound.
- To examine comms overhead, try small models (flops-scale < 1) then larger ones.
- For a single-GPU baseline, use --nproc_per_node=1.
"""

import argparse, os, time, math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class SyntheticImages(Dataset):
    def __init__(self, num_samples=20000, shape=(3,224,224), num_classes=1000):
        self.num_samples = num_samples
        self.shape = shape
        self.num_classes = num_classes
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        x = torch.randn(*self.shape, dtype=torch.float32)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y

def make_model(kind="conv", flops_scale=1.0, num_classes=1000):
    if kind == "mlp":
        hidden = int(2048 * flops_scale)
        return torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3*224*224, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, num_classes),
        )
    else:  # conv
        c = max(8, int(32 * flops_scale))
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, c, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(c, 2*c, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(2*c, 4*c, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Flatten(),
            torch.nn.Linear(4*c, num_classes)
        )

def setup(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def run_ddp(rank, world_size, args):
    setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")
    model = make_model(args.model, args.flops_scale, args.num_classes).to(device)
    model = DDP(model, device_ids=[rank])

    dataset = SyntheticImages(num_samples=args.samples, shape=(3,224,224), num_classes=args.num_classes)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)

    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Warmup
    torch.cuda.synchronize(device)
    for _ in range(2):
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()
            break

    # Timed epochs
    epoch_times = []
    total_samples = 0
    for ep in range(args.epochs):
        sampler.set_epoch(ep)
        t0 = time.time()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()
            total_samples += xb.shape[0] * world_size  # global samples progressed
        torch.cuda.synchronize(device)
        dt = time.time() - t0
        epoch_times.append(dt)

    # Aggregate epoch time (mean) from all ranks to rank0
    t = torch.tensor(sum(epoch_times)/len(epoch_times), dtype=torch.float32, device=device)
    dist.reduce(t, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        avg_epoch_time = (t.item() / world_size)
        throughput = (args.samples // args.batch_size) * args.batch_size / avg_epoch_time  # per-epoch samples / time
        print(f"RESULT world_size={world_size} model={args.model} flops_scale={args.flops_scale} "
              f"batch={args.batch_size} epochs={args.epochs} "
              f"avg_epoch_s={avg_epoch_time:.3f} samples_per_s={throughput:.2f}")
    cleanup()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--samples", type=int, default=20000, help="global samples per epoch")
    ap.add_argument("--model", type=str, default="conv", choices=["conv","mlp"])
    ap.add_argument("--flops-scale", type=float, default=1.0)
    ap.add_argument("--num-classes", type=int, default=1000)
    args = ap.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    # torchrun launches set WORLD_SIZE and RANK; SLURM+torchrun is recommended
    if world_size == 1:
        # Single GPU (no DDP init) for simplest baseline
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = make_model(args.model, args.flops_scale, args.num_classes).to(device)
        dataset = SyntheticImages(num_samples=args.samples, shape=(3,224,224), num_classes=args.num_classes)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()
        # Warmup
        for _ in range(2):
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                yhat = model(xb)
                loss = loss_fn(yhat, yb)
                loss.backward()
                opt.step()
                break
        # Timed
        t0 = time.time()
        for ep in range(args.epochs):
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                yhat = model(xb)
                loss = loss_fn(yhat, yb)
                loss.backward()
                opt.step()
        torch.cuda.synchronize() if device.type == "cuda" else None
        dt = time.time() - t0
        avg_epoch = dt / args.epochs
        throughput = (args.samples // args.batch_size) * args.batch_size / avg_epoch
        print(f"RESULT world_size=1 model={args.model} flops_scale={args.flops_scale} "
              f"batch={args.batch_size} epochs={args.epochs} "
              f"avg_epoch_s={avg_epoch:.3f} samples_per_s={throughput:.2f}")
    else:
        run_ddp(rank, world_size, args)

if __name__ == "__main__":
    main()
