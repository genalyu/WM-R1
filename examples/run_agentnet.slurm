#!/bin/bash
#SBATCH --job-name=wm_r1
#SBATCH --partition=a100          # 根据你的集群分区名修改
#SBATCH --nodes=1                 # 运行节点数
#SBATCH --ntasks-per-node=1       # 每个节点的任务数
#SBATCH --gres=gpu:2              # 每个节点使用的 GPU 数量
#SBATCH --cpus-per-task=32        # 每个任务分配的 CPU 核心数
#SBATCH --mem=256G                # 内存分配
#SBATCH --output=logs/agentnet_%j.out
#SBATCH --error=logs/agentnet_%j.err

# --- 1. 环境初始化 ---
source ~/.bashrc
# 显式加载 Conda (请根据你的 anaconda/miniconda 路径修改)
source ~/jyy/anaconda/etc/profile.d/conda.sh 
conda activate 3spo

# 确保在项目根目录
cd $(dirname $(dirname $(realpath $0)))
export PYTHONPATH=$PYTHONPATH:$(pwd)

# --- 2. 配置实验参数 ---
EXPERIMENT_NAME="wm_r1"
MODEL_PATH="/public/home/xlwang/genalyu/models/Qwen2.5-VL-3B-Instruct"
SYSTEM_PROMPT="You are a helpful assistant."

# 根据 Slurm 资源自动获取或手动指定
NUM_GPUS=2
NUM_ENVS=8
ROLLOUT_N=6

# 数据集路径 (请修改为你的实际路径)
TRAIN_FILES="/public/home/xlwang/genalyu/dataset/agentnet/train.jsonl"
VAL_FILES="/public/home/xlwang/genalyu/dataset/agentnet/val.jsonl"

# --- 3. 启动 Ray (单节点模式) ---
# 在单节点 Slurm 任务中，verl.trainer.main 会自动初始化 Ray。
# 如果是多节点，需要先在各节点启动 ray start。

# --- 4. 运行训练命令 ---
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.format_prompt="${SYSTEM_PROMPT}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.max_prompt_length=64000 \
    data.max_response_length=8192 \
    data.max_pixels=2116800 \
    data.min_pixels=256 \
    data.rollout_batch_size=2 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.max_grad_norm=1.0 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.ulysses_sequence_parallel_size=1 \
    worker.actor.padding_free=true \
    worker.actor.ppo_epochs=1 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.3 \
    worker.actor.global_batch_size=1 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.gpu_memory_utilization=0.6 \
    worker.rollout.temperature=1.0 \
    worker.rollout.n=$ROLLOUT_N \
    worker.rollout.limit_images=15 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_num_batched_tokens=128000 \
    algorithm.disable_kl=True \
    algorithm.kl_coef=0 \
    algorithm.enable_replay=False \
    env.num_envs=$NUM_ENVS \
    env.max_steps=15 \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=16 \
    trainer.save_limit=3 \
    trainer.val_before_train=True \
    trainer.val_freq=16 \
    trainer.total_episodes=15
