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

# --- WandB 配置 ---
export WANDB_PROJECT="wm-r1"   # 你在 WandB 上的项目名称
export WANDB_MODE="offline"             # 离线模式，不连接网络
# export WANDB_API_KEY="你的_WANDB_API_KEY" # 离线模式通常不需要 API Key
# export WANDB_ENTITY="你的_ENTITY"

# 确保在项目根目录
cd $(dirname $(dirname $(realpath $0)))
export PYTHONPATH=$PYTHONPATH:$(pwd)/Code2World

# --- 2. 配置实验参数 ---
EXPERIMENT_NAME="wm_r1"
MODEL_PATH="/public/home/xlwang/genalyu/models/Qwen2.5-VL-3B-Instruct"
SYSTEM_PROMPT="You are a helpful assistant."

# 本地 World Model 配置 (通过 vLLM 启动的本地服务)
WM_MODEL_PATH="/public/home/xlwang/genalyu/models/code2world"
WM_MODEL_NAME="code2world"
WM_API_BASE="http://127.0.0.1:9000/v1/chat/completions" # 指向本地 vLLM 端口
WM_API_KEY="EMPTY" # 本地服务通常不需要 key

# 根据 Slurm 资源自动获取或手动指定
NUM_GPUS=2
NUM_ENVS=8
ROLLOUT_N=6

# 数据集路径 (请修改为你的实际路径)
TRAIN_FILES="/public/home/xlwang/genalyu/dataset/agentnet/agentnet_train_9k.jsonl"
VAL_FILES="/public/home/xlwang/genalyu/dataset/agentnet/agentnet_test_1k.jsonl"

# --- 3. 启动本地 World Model 服务 (vLLM) ---
echo "Starting local World Model server..."
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --served-model-name ${WM_MODEL_NAME} \
    --model ${WM_MODEL_PATH} \
    --trust-remote-code \
    --limit-mm-per-prompt image=16 \
    --max-model-len 32768 \
    --port 9000 &
WM_PID=$!

# 等待服务就绪
echo "Waiting for World Model server to be ready..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:9000/v1/models > /dev/null; then
        echo "World Model server is READY!"
        break
    fi
    sleep 10
done

# --- 4. 运行训练命令 (展平所有配置，不依赖 config.yaml) ---
python3 -m verl.trainer.main \
    trainer.project_name="wm-r1-agentnet" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="['console', 'wandb']" \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.default_local_dir=/public/home/xlwang/genalyu/WM-R1/checkpoints/$SLURM_JOB_ID \
    trainer.save_freq=16 \
    trainer.save_limit=3 \
    trainer.val_before_train=True \
    trainer.val_freq=16 \
    trainer.total_episodes=15 \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.format_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=64000 \
    data.max_response_length=8192 \
    data.max_pixels=2116800 \
    data.min_pixels=256 \
    data.rollout_batch_size=2 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.ppo_epochs=1 \
    worker.actor.global_batch_size=1 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.n=$ROLLOUT_N \
    worker.rollout.temperature=1.0 \
    worker.rollout.tensor_parallel_size=1 \
    algorithm.adv_estimator="wm_r1" \
    algorithm.disable_kl=True \
    env.num_envs=$NUM_ENVS \
    env.max_steps=15 \
    env.wm_model_name="${WM_MODEL_NAME}" \
    env.wm_api_base="${WM_API_BASE}" \
    env.wm_api_key="${WM_API_KEY}"

# --- 5. 清理 ---
kill $WM_PID
