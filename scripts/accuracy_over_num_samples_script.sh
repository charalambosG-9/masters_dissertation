#!/bin/bash   
#SBATCH --partition=gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=20:30:00  
#SBATCH --output=./outputs/accuracy_over_num_samples_output.txt  

module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate dissenv

python ./code/myconfig.py
python  -u ./code/accuracy_over_num_samples.py 