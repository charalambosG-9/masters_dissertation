#!/bin/bash   
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=20:30:00 
#SBATCH --output=./outputs/unzip_files_output.txt 

module load Anaconda3/2022.05

source activate dissenv

python ./code/unzip_files.py 