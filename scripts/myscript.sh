#!/bin/bash
#SBATCH -p veu             # Partition to submit to
#SBATCH --mem=30G      # Max CPU Memory
#SBATCH --gres=gpu:1
python test.py