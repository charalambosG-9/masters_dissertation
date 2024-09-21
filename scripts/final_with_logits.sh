#!/bin/bash   
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=02:00:00  
#SBATCH --output=./outputs/final_with_logits_output.txt  

module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate dissenv

python ./code/myconfig.py
python -u ./code/final_with_logits.py 