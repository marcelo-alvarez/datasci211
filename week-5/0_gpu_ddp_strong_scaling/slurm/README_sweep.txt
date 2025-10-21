# Example submission sequence (sweep 1,2,4,8 GPUs on a single node):
sbatch --gres=gpu:1 slurm/ddp_single_node.sbatch   # baseline
sbatch --gres=gpu:2 slurm/ddp_single_node.sbatch
sbatch --gres=gpu:4 slurm/ddp_single_node.sbatch
sbatch --gres=gpu:8 slurm/ddp_single_node.sbatch

# After runs finish, aggregate:
python python/aggregate_scaling.py ddp_<job1>.out ddp_<job2>.out ddp_<job3>.out ddp_<job4>.out --out scaling.png
