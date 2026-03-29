#!/bin/bash
#SBATCH --job-name=3spo_setup
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=setup_%j.out
#SBATCH --time=01:00
source ~/.bashrc
conda activate 3spo

cd /public/home/xlwang/genalyu/3SPO/
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
cd OSWorld
pip install -r requirements.txt
