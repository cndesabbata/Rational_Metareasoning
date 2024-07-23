#!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --constraint=gpu80       # gpu model   
#SBATCH --mem-per-cpu=64G        # total memory (RAM) per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2024.2
conda activate thesis
echo $PATH
cd src/
python train.py