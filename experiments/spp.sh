#!/bin/bash -e
#SBATCH --job-name=gswg    
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-gpu=125G
#SBATCH --cpus-per-gpu=32
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.TrangNVT2@vinai.io

module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/vinai/users/trangnvt2/sw

experiment=no_bn_no_norm

for d_lr in 0.00005 0.0001 0.00001 0.0003 
do
for clip in 0.01 0.005 0.05 10
do 
for generator_steps in -10 1 3 
do

python sinkhorn-gan/run_sinkhorn.py --dataset cifar10 --dataroot data/ --lr_d $d_lr --clip $clip --generator_steps $generator_steps --experiment $experiment

python sinkhorn-gan/run_sinkhorn.py --dataset cifar10 --dataroot data/ --lr_d $d_lr --lr_g $d_lr --clip $clip --generator_steps $generator_steps --experiment $experiment


done
done
done

