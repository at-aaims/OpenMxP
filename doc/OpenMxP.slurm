#!/bin/bash
#SBATCH -A stf018
#SBATCH -J openmxp
#SBATCH -p batch

#SBATCH -N 32

##SBATCH -C nvme

#SBATCH -t 0:30:00

#SBATCH -o frontier_OpenMxP_01.out

module load PrgEnv-gnu/8.3.3
module load gcc

module load rocm/5.1.0
module load cray-mpich/8.1.18
module load craype-x86-trento
module load craype-network-ofi
module load craype-accel-host

export LD_LIBRARY_PATH=/opt/rocm-5.1.0/llvm/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/rocm-5.1.0/lib:${LD_LIBRARY_PATH}

#export LD_LIBRARY_PATH=/opt/rocm-5.4.0/llvm/lib:${LD_LIBRARY_PATH}
#export LD_LIBRARY_PATH=/opt/rocm-5.4.0/lib:${LD_LIBRARY_PATH}

export MPIR_CVAR_GPU_EAGER_DEVICE_MEM=0
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_SMP_SINGLE_COPY_MODE=CMA

echo "JobID          : $SLURM_JOB_ID"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"

#SLURM_JOB_NODELIST SLURM_JOB_ID
#echo $SLURM_JOB_NODELIST > "frontier_$SLURM_JOB_ID.nodes"
#echo $SLURM_JOB_NODELIST

nt=$(expr $SLURM_JOB_NUM_NODES \* 8)

pq=24
pq=28
pq=32

pq=16

comm=4

b=2560

ln=92160
ln=120320
ln=122880
ln=125440

N=$(expr $pq \* $ln)

export OMP_NUM_THREADS=7

export NOTE="OpenMxP rocm 5.1.0 GPU-Direct"

export CMD="../build/OpenMxP.x86_64 $N $b $pq -1 -comm 2 -alt 1 "
srun -N $SLURM_JOB_NUM_NODES -n $nt -c 7 --ntasks-per-node=8 --gpus-per-task=1 --gpu-bind=closest $CMD

export CMD="../build/OpenMxP.x86_64 $N $b $pq -1 -comm 2 -alt 2 "
srun -N $SLURM_JOB_NUM_NODES -n $nt -c 7 --ntasks-per-node=8 --gpus-per-task=1 --gpu-bind=closest $CMD

