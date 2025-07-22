#!/bin/bash
#SBATCH --job-name=fsdp-benchmark
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=96
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --time=01:30:00

nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Head node IP: $head_node_ip"

module load python/3.12.1-gpu
source /work/tc067/tc067/s2638244/msc-cirkit/myvenv/bin/activate

export WANDB_MODE=offline
export WANDB_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXX

srun torchrun \
  --nnodes $SLURM_JOB_NUM_NODES \
  --nproc_per_node 1 \
  --rdzv_id $RANDOM \
  --rdzv_backend c10d \
  --rdzv_endpoint $head_node_ip:29500 \
  experiments/wand_benchmark.py --distributed fsdp --powers-of-two \
      --min-exp 5 --max-exp 7 --circuit-structure deep_cp_circuit --depth 5
