#!/bin/bash
#SBATCH --job-name=fsdp-benchmark
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:2            # request 2 GPUs
#SBATCH --account=tc067-s2638244
#SBATCH --time=01:30:00
#SBATCH --account=tc067-s2638244

export WANDB_MODE=offline

export MPLCONFIGDIR="/work/tc067/tc067/s2638244/.config/matplotlib"

# create it (race-safe)
mkdir -p "$MPLCONFIGDIR"

export PYTORCH_KERNEL_CACHE_PATH=/work/tc067/tc067/s2638244/.config/torch_kernels
mkdir -p "$PYTORCH_KERNEL_CACHE_PATH"


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


# ENABLE BELOW FOR DEBUGGING
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL        
# export NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1           
# export TORCH_NCCL_BLOCKING_WAIT=1                  # watchdog in seconds, not minutes
# export TORCH_NCCL_TRACE_BUFFER_SIZE=$((32*1024*1024))
# export TORCH_NCCL_DUMP_ON_TIMEOUT=1   

# Single‐node, multi‐GPU launch
srun -n 1 torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  experiments/wand_benchmark.py \
    --distributed fsdp \
    --powers-of-two \
    --min-exp 5 --max-exp 7 \
    --circuit-structure deep_cp_circuit \
    --depth 5
