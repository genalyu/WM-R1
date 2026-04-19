#!/bin/bash
#SBATCH --job-name=wm_r1_ablation_wm_no_think
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --output=logs/ablation_wm_no_think_%j.out
#SBATCH --error=logs/ablation_wm_no_think_%j.err

# ============================================================
# Ablation: 使用世界模型做状态预测，但禁用 <think> 中的深度思考
# GPU 分配：
#   GPU 0 → vLLM WM 服务
#   GPU 1 → 训练 (FSDP + vLLM rollout)
# ============================================================

# --- 1. 环境初始化 ---
source ~/.bashrc
source ~/jyy/anaconda/etc/profile.d/conda.sh
conda activate 3spo

# 清理所有残留 Ray 状态
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray
rm -rf ~/.config/ray/
pkill -9 -f raylet 2>/dev/null || true
pkill -9 -f gcs_server 2>/dev/null || true
sleep 2

# --- 全局环境变量 ---
export WANDB_PROJECT="wm-r1"
export WANDB_MODE="offline"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=0

cd $(dirname $(dirname $(realpath $0)))
PROJECT_DIR=$(pwd)
export PROJECT_DIR
export WANDB_DIR=/public/home/xlwang/genalyu/WM-R1

# World Model 配置
WM_MODEL_PATH="/public/home/xlwang/genalyu/models/code2world"
WM_PORT=18889

# --- 每节点启动 vLLM WM 服务 (GPU 0) ---
echo "Starting WM vLLM server on $(hostname) GPU:0, port $WM_PORT"
export CONDA_PREFIX=""
eval "$(conda shell.bash hook)"
conda activate wm
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "$WM_MODEL_PATH" \
    --host 0.0.0.0 \
    --port $WM_PORT \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code &
sleep 10
export WM_HTTP_URL="http://$(hostname -I | awk '{print $1}'):${WM_PORT}"
echo "WM server URL: $WM_HTTP_URL"

# 等 WM server 真正可用
for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:${WM_PORT}/v1/models > /dev/null 2>&1; then
        echo "WM server ready after ${i}s"
        break
    fi
    sleep 1
done
HEAD_NODE_IP=$(hostname -I | awk '{print $1}')
export HEAD_NODE_IP
export VLLM_HOST_IP=$HEAD_NODE_IP

# 启动 Ray Head (GPU 1 only, GPU 0 is for WM vLLM)
echo "Starting Ray HEAD on $(hostname) ($HEAD_NODE_IP)"
export CUDA_VISIBLE_DEVICES=1
nohup ray start --head --node-ip-address=$HEAD_NODE_IP --port=6379 \
    --num-gpus=1 --resources="{\"docker:$HEAD_NODE_IP\": 128}" \
    > /tmp/ray_head.log 2>&1 &

echo "Waiting for GCS to become available..."
for i in $(seq 1 60); do
    if ray healthcheck 2>/dev/null; then
        echo "GCS ready after ${i}s"
        break
    fi
    sleep 1
done

# 其他节点启动 Ray Worker + WM vLLM 服务
sleep 5
if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
    echo "Starting Ray WORKERS + WM servers on other nodes"
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
            export TOKENIZERS_PARALLELISM=false
            export OMP_NUM_THREADS=4
            export HEAD_NODE_IP='"$HEAD_NODE_IP"'
            export VLLM_HOST_IP=$HEAD_NODE_IP
            export PYTHONPATH=$PYTHONPATH:'"$PROJECT_DIR"'/Code2World

            # WM vLLM 服务 (GPU 0)
            WM_PORT=18889
            echo "Starting WM vLLM server on $(hostname) GPU:0"
            export CONDA_PREFIX=""
            eval "$(conda shell.bash hook)"
            conda activate wm
            CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
                --model '"$WM_MODEL_PATH"' \
                --host 0.0.0.0 \
                --port $WM_PORT \
                --tensor-parallel-size 1 \
                --max-model-len 8192 \
                --gpu-memory-utilization 0.9 \
                --trust-remote-code &
            sleep 10

            # Wait for WM server to become available
            echo "Waiting for WM server on $(hostname)..."
            for i in $(seq 1 120); do
                if curl -sf http://127.0.0.1:${WM_PORT}/v1/models > /dev/null 2>&1; then
                    echo "WM server on $(hostname) ready after ${i}s"
                    break
                fi
                sleep 1
            done
            # Ray Worker (GPU 1 only, GPU 0 is for WM vLLM)
            ray stop --force 2>/dev/null || true
            rm -rf /tmp/ray
            rm -rf ~/.config/ray/
            pkill -9 -f raylet 2>/dev/null || true
            pkill -9 -f gcs_server 2>/dev/null || true
            sleep 2
            export CUDA_VISIBLE_DEVICES=1
            nohup ray start --address=$HEAD_NODE_IP:6379 --num-gpus=1 --resources="{\"docker:$(hostname -I | awk \"{print \$1}\")\": 128}" \
                > /tmp/ray_worker.log 2>&1 &
            disown
            echo "Ray worker started on $(hostname)"
        '
    echo "All worker nodes initialized"
fi

# --- 2. 配置 ---
EXPERIMENT_NAME="ablation_wm_no_think"
MODEL_PATH="/public/home/xlwang/genalyu/models/Qwen2.5-VL-3B-Instruct"
SYSTEM_PROMPT="You are a helpful assistant."

NUM_GPUS_PER_NODE=1
NUM_NODES=$SLURM_JOB_NUM_NODES
ROLLOUT_N=$((3 * NUM_NODES))
NUM_ENVS=$((2 * ROLLOUT_N))  # env workers 必须等于 rollout_batch_size * rollout_n

TRAIN_FILES="/public/home/xlwang/genalyu/dataset/agentnet/agentnet_train_9k.jsonl"
VAL_FILES="/public/home/xlwang/genalyu/dataset/agentnet/agentnet_test_1k.jsonl"

# --- 等待所有 Ray 节点 GPU 就绪 ---
EXPECTED_GPU=$((NUM_NODES * NUM_GPUS_PER_NODE))
echo "Waiting for Ray cluster: expecting $EXPECTED_GPU GPUs..."
for i in $(seq 1 120); do
    ray status > /tmp/ray_status.txt 2>/dev/null
    if [ $? -eq 0 ]; then
        # Parse GPU count from ray status output
        # Example: "GPU (0.0-1.0): 2.0/2.0 used"
        AVAILABLE_GPU=$(grep "^GPU" /tmp/ray_status.txt | head -1 | grep -oP '\d+\.0(?=/)' | head -1 | cut -d'.' -f1)
        if [ -z "$AVAILABLE_GPU" ]; then AVAILABLE_GPU=0; fi
    fi
    if [ "$AVAILABLE_GPU" -ge "$EXPECTED_GPU" ]; then
        echo "Ray cluster ready: $AVAILABLE_GPU GPUs after ${i}s"
        break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "  ${i}s: only $AVAILABLE_GPU/$EXPECTED_GPU GPUs ready..."
    fi
    sleep 1
done
if [ "$AVAILABLE_GPU" -lt "$EXPECTED_GPU" ]; then
    echo "ERROR: Only $AVAILABLE_GPU/$EXPECTED_GPU GPUs available after 120s. Worker nodes may not have joined."
    exit 1
fi

# --- 3. 训练 (GPU 1, HTTP 连接 WM) ---
cd $PROJECT_DIR
conda activate 3spo
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/Code2World

# NCCL 跨节点通信配置
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0,ib0,eno,ens,enp
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=600

# 限制线程数，避免 Ray worker 过多导致资源竞争
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# Fix GCC/linker mismatch for Triton JIT compilation
export LD_LIBRARY_PATH=/public/software/compiler/gnu/gcc-14.3.0/lib64:${LD_LIBRARY_PATH:-}
export CC=/public/software/compiler/gnu/gcc-14.3.0/bin/gcc
export AR=/public/software/compiler/gnu/gcc-14.3.0/bin/ar
export CUDA_HOST_COMPILER=$CC

python3 -m verl.trainer.main \
    trainer.project_name="wm-r1-ablation" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="['console', 'wandb']" \
    trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.nnodes=$NUM_NODES \
    trainer.default_local_dir=/public/home/xlwang/genalyu/WM-R1/checkpoints/ablation_wm_no_think/$SLURM_JOB_ID \
    trainer.save_freq=16 \
    trainer.save_limit=3 \
    trainer.val_before_train=True \
    trainer.val_freq=16 \
    trainer.total_episodes=15 \
    env.use_wm=True \
    env.use_local_wm=False \
    env.wm_http_url="${WM_HTTP_URL}" \
    env.n_wm_max=0 \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.format_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=8192 \
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
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.n=$ROLLOUT_N \
    worker.rollout.name="vllm" \
    worker.rollout.temperature=1.0 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.limit_images=16 \
    worker.rollout.max_num_batched_tokens=9000 \
    algorithm.adv_estimator="wm_r1" \
    algorithm.disable_kl=True \
    env.num_envs=$NUM_ENVS \
    env.max_steps=15 \
    data.num_workers=4

# --- 5. 清理 ---
ray stop
