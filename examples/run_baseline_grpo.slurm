#!/bin/bash
#SBATCH --job-name=wm_r1_baseline
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=330G
#SBATCH --output=logs/baseline_grpo_%j.out
#SBATCH --error=logs/baseline_grpo_%j.err

# ============================================================
# Baseline: Pure GRPO — 直接生成完整轨迹，不使用世界模型
# ============================================================

# --- 1. 环境初始化 ---
source ~/.bashrc
source ~/jyy/anaconda/etc/profile.d/conda.sh
conda activate 3spo

# --- 全局环境变量 (必须在 Ray 启动前设置) ---
export WANDB_PROJECT="wm-r1"
export WANDB_MODE="offline"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=0

# 确保在项目根目录
cd $(dirname $(dirname $(realpath $0)))
PROJECT_DIR=$(pwd)
export PROJECT_DIR
mkdir -p $PROJECT_DIR/wandb
export WANDB_DIR=$PROJECT_DIR/wandb

# --- 启动 Ray 集群 (多节点支持) ---
# 先清理残留进程
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray

# --- 启动 Ray 集群 (多节点支持) ---
HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
export HEAD_NODE_IP
export VLLM_HOST_IP=$HEAD_NODE_IP

echo "Starting Ray HEAD on $(hostname) ($HEAD_NODE_IP)"
ray start --head --node-ip-address=$HEAD_NODE_IP --port=6379 \
    --resources="{\"docker:$HEAD_NODE_IP\": 128}" --block &

# 等待 GCS 就绪 (主动探测，不等死 sleep)
echo "Waiting for GCS to become available..."
for i in $(seq 1 60); do
    if ray healthcheck 2>/dev/null; then
        echo "GCS ready after ${i}s"
        break
    fi
    sleep 1
done

sleep 5
if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
    echo "Starting Ray WORKERS on other nodes"
    srun --nodes=$((SLURM_JOB_NUM_NODES - 1)) --ntasks=$((SLURM_JOB_NUM_NODES - 1)) --exclude=$(hostname) \
        bash -c '
            source ~/.bashrc
            source ~/jyy/anaconda/etc/profile.d/conda.sh
            conda activate 3spo
            export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
            export CUDA_DEVICE_ORDER=PCI_BUS_ID
            export VLLM_ATTENTION_BACKEND=XFORMERS
            export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
            export TORCH_COMPILE_DISABLE=1
            export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=0
            export HEAD_NODE_IP='"$HEAD_NODE_IP"'
            export VLLM_HOST_IP=$HEAD_NODE_IP
            export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/Code2World
            ray stop --force 2>/dev/null || true
            rm -rf /tmp/ray
            ray start --address=$HEAD_NODE_IP:6379 --resources="{\"docker:$(hostname -I | awk \"{print \$1}\")\": 128}" --block
        ' &
    sleep 10
fi

# --- 2. 配置实验参数 ---
EXPERIMENT_NAME="baseline_grpo"
MODEL_PATH="/public/home/xlwang/genalyu/models/Qwen2.5-VL-3B-Instruct"
SYSTEM_PROMPT="You are a helpful assistant."

# 根据 Slurm 资源自动获取
NUM_GPUS_PER_NODE=2
NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_ENVS=$((1 * NUM_NODES))
ROLLOUT_N=$((3 * NUM_NODES))

# 数据集路径
TRAIN_FILES="/public/home/xlwang/genalyu/dataset/agentnet/agentnet_train_9k.jsonl"
VAL_FILES="/public/home/xlwang/genalyu/dataset/agentnet/agentnet_test_1k.jsonl"

# --- 3. 运行训练命令 (Baseline: env.use_wm=False, algorithm.adv_estimator="grpo") ---
cd $PROJECT_DIR
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/Code2World
python3 -m verl.trainer.main \
    trainer.project_name="wm-r1-baseline" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="['console', 'wandb']" \
    trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.nnodes=$NUM_NODES \
    trainer.default_local_dir=/public/home/xlwang/genalyu/WM-R1/checkpoints/baseline_grpo/$SLURM_JOB_ID \
    trainer.save_freq=16 \
    trainer.save_limit=3 \
    trainer.val_before_train=True \
    trainer.val_freq=16 \
    trainer.total_episodes=15 \
    env.use_wm=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.format_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
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
    worker.actor.use_torch_compile=False \
    worker.rollout.gpu_memory_utilization=0.7 \
    worker.rollout.n=$ROLLOUT_N \
    worker.rollout.name="vllm" \
    worker.rollout.temperature=1.0 \
    worker.rollout.tensor_parallel_size=$NUM_GPUS_PER_NODE \
    worker.rollout.max_num_batched_tokens=5120 \
    algorithm.adv_estimator="grpo" \
    algorithm.disable_kl=True \
    env.num_envs=$NUM_ENVS \
    env.max_steps=15

# --- 5. 清理 ---
ray stop
