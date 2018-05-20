#!/bin/sh
#
#SBATCH --verbose
#SBATCH --job-name=chest-xray
#SBATCH --output=./log/slurm_%j.out
#SBATCH --error=./log/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --mem=100GB
#SBATCH --gres=gpu:p40:1
#SBATCH --mail-type=END
#SBATCH --mail-user=ma2510@nyu.edu

module purge
# source activate /home/$USER/.conda/envs/tensorflow_env
module load numpy/intel/1.13.1
module load scikit-image/intel/0.13.1
module load scipy/intel/0.19.1
module load scikit-learn/intel/0.18.1
module load tensorflow/python2.7/1.3.0

cd /scratch/ma2510/dsh/xray-ensemble

python src/main.py
