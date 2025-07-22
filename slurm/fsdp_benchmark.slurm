#!/bin/bash
#SBATCH --job-name=fsdp-benchmark
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2            # request 2 GPUs
#SBATCH --account=tc067-s2638244
#SBATCH --time=01:30:00
#SBATCH --account=tc067-s2638244

export WANDB_MODE=offline

module load python/3.12.1-gpu
source /work/tc067/tc067/s2638244/msc-cirkit/myvenv/bin/activate

# Echo out the Slurm allocation
echo "Allocated nodes:    $SLURM_JOB_NODELIST"
echo "Total nodes:        $SLURM_NNODES"
echo "Total tasks:        $SLURM_NTASKS"
echo "Tasks per node:     $SLURM_NTASKS_PER_NODE"
echo "CPUs per task:      $SLURM_CPUS_PER_TASK"
env | grep ^SLURM | sort
# Set OMP threads to avoid oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Offline logging for wandb
export WANDB_MODE=offline
export WANDB_API_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Single‐node, multi‐GPU launch
srun torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  experiments/wand_benchmark.py \
    --distributed fsdp \
    --powers-of-two \
    --min-exp 5 --max-exp 7 \
    --circuit-structure deep_cp_circuit \
    --depth 5
