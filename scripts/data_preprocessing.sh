#!/bin/bash   
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=100G
#SBATCH --time=20:30:00 
#SBATCH --output=./outputs/data_preprocessing_output.txt  

module load Anaconda3/2022.05

source activate dissenv

python ./code/myconfig.py
python ./code/data_preprocessing.py 