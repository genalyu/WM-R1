#!/bin/bash
#SBATCH --job-name=wm_r1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/agentnet_%j.out
#SBATCH --error=logs/agentnet_%j.err

# ============================================================
# WM-R1 主实验：使用 World Model 进行 RL 训练
# 单卡模式：WM 服务和训练共享同一张 GPU
# ============================================================

# --- 用户配置区（按需修改） ---
CONDA_ENV="wmr1"                    # 训练环境
CONDA_WM_ENV="wmr1"                 # WM 服务环境（可和训练同一个，也可分开）
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"   # Agent 策略模型
WM_MODEL_PATH="Qwen/Qwen3-VL-8B"           # World Model
TRAIN_FILES="./data/agentnet_train.jsonl"
VAL_FILES="./data/agentnet_test.jsonl"

# --- 项目根目录（自动检测，不需要改） ---
PROJECT_DIR="$(cd "$(dirname "$(dirname "$(realpath "$0")")")" && pwd)"
cd "$PROJECT_DIR"

# --- 环境初始化 ---
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# 清理残留 Ray 状态
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray ~/.config/ray/
pkill -9 -f raylet 2>/dev/null || true
pkill -9 -f gcs_server 2>/dev/null || true
fuser -k 6379/tcp 2>/dev/null || true
pkill -9 -f redis 2>/dev/null || true
sleep 2

# --- 全局环境变量 ---
export WANDB_PROJECT="wm-r1"
export WANDB_MODE="offline"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONPATH="$PYTHONPATH:$PROJECT_DIR/Code2World"

# ============================================================
# 启动 World Model 服务
# ============================================================
WM_PORT=18888
echo "Starting WM server on GPU:0, port $WM_PORT"

conda activate "$CONDA_WM_ENV"
CUDA_VISIBLE_DEVICES=0 python "$PROJECT_DIR/Code2World/android_world/agents/wm_server.py" \
    --model "$WM_MODEL_PATH" \
    --port "$WM_PORT" \
    --device cuda:0 &
WM_PID=$!
sleep 8

export WM_HTTP_URL="http://127.0.0.1:${WM_PORT}"

# 等 WM server 可用
echo "Waiting for WM server..."
for i in $(seq 1 120); do
    if curl -sf "$WM_HTTP_URL/v1/models" > /dev/null 2>&1; then
        echo "WM server ready after ${i}s"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "ERROR: WM server failed to start after 120s"
        exit 1
    fi
    sleep 1
done

# 切回训练环境
conda activate "$CONDA_ENV"

# ============================================================
# 启动 Ray（单节点）
# ============================================================
HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
export HEAD_NODE_IP
export RAY_ADDRESS="$HEAD_NODE_IP:6379"

echo "Starting Ray HEAD on $(hostname) ($HEAD_NODE_IP)"
nohup ray start --head --node-ip-address="$HEAD_NODE_IP" --port=6379 \
    --num-gpus=1 > /tmp/ray_head.log 2>&1 &

sleep 3

# 等 GCS 就绪
echo "Waiting for Ray GCS..."
for i in $(seq 1 120); do
    if bash -c 'echo > /dev/tcp/127.0.0.1/6379' 2>/dev/null; then
        echo "GCS ready after ${i}s"
        break
    fi
    sleep 1
done
sleep 5

# ============================================================
# 训练配置
# ============================================================
EXPERIMENT_NAME="wm_r1_android"
NUM_GPUS_PER_NODE=1
ROLLOUT_N=2
NUM_ENVS=$((2 * ROLLOUT_N))  # = rollout_batch_size * rollout_n

# ============================================================
# 启动训练
# ============================================================
python3 -m verl.trainer.main \
    trainer.project_name="wm-r1-android" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="['console', 'wandb']" \
    trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=16 \
    trainer.save_limit=3 \
    trainer.val_before_train=False \
    trainer.val_freq=999999 \
    trainer.total_episodes=15 \
    \
    env.platform="android" \
    env.use_wm=True \
    env.use_local_wm=False \
    env.wm_http_url="${WM_HTTP_URL}" \
    env.num_envs=$NUM_ENVS \
    env.max_steps=15 \
    env.n_wm_max=5 \
    \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.format_prompt="You are a helpful assistant." \
    data.max_prompt_length=28000 \
    data.max_response_length=2048 \
    data.max_pixels=4194304 \
    data.min_pixels=256 \
    data.rollout_batch_size=2 \
    data.num_workers=2 \
    data.max_trajectory_steps=3 \
    \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.ppo_epochs=1 \
    worker.actor.global_batch_size=2 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.use_torch_compile=False \
    \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.n=$ROLLOUT_N \
    worker.rollout.name="vllm" \
    worker.rollout.temperature=1.0 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.limit_images=16 \
    worker.rollout.max_num_batched_tokens=32000 \
    \
    algorithm.adv_estimator="wm_r1" \
    algorithm.disable_kl=True \
    \
    reward.alpha=1.0 \
    reward.beta=0.5 \
    data.val_batch_size=2

# --- 清理 ---
ray stop
kill $WM_PID 2>/dev/null || true
echo "Training finished."
