#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --time=0-01:00:00
#SBATCH --output=notebook.out

module load Anaconda3
jupyter notebook --no-browser --ip='*' --port=7777
