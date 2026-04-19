#!/bin/bash
#SBATCH --job-name=debug_img
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/debug_img_%j.out
#SBATCH --error=logs/debug_img_%j.err
#SBATCH --time=00:30:00

source ~/.bashrc
source ~/jyy/anaconda/etc/profile.d/conda.sh
conda activate 3spo

cd /public/home/xlwang/genalyu/WM-R1

python debug_image_mismatch.py \
    --data-path /public/home/xlwang/genalyu/dataset/agentnet/agentnet_test_1k.jsonl \
    --model-path /public/home/xlwang/genalyu/models/Qwen2.5-VL-3B-Instruct \
    --image-dir /public/home/xlwang/genalyu/dataset/agentnet/ \
    --num-samples 10 \
    --max-pixels 2116800 \
    --min-pixels 256
