#!/bin/bash
#SBATCH --job-name=wm_r1
#SBATCH --partition=a100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=330G
#SBATCH --output=logs/agentnet_%j.out
#SBATCH --error=logs/agentnet_%j.err

# ============================================================
# GPU 分配策略：
#   GPU 0 → vLLM WM 服务 (每个节点各起一个)
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
# Kill any leftover Redis that might hold port 6379
fuser -k 6379/tcp 2>/dev/null || true
pkill -9 -f redis 2>/dev/null || true
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

cd /public/home/xlwang/genalyu/WM-R1
PROJECT_DIR="/public/home/xlwang/genalyu/WM-R1"
export PROJECT_DIR
export WANDB_DIR=/public/home/xlwang/genalyu/WM-R1/

# World Model 配置
WM_MODEL_PATH="/public/home/xlwang/genalyu/models/code2world"
WM_PORT=18888

# --- 每节点启动 WM 服务 (GPU 0) — 使用 transformers 原生加载 Qwen3-VL ---
echo "Starting WM server on $(hostname) GPU:0, port $WM_PORT"
export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/Code2World
export CONDA_PREFIX=""
eval "$(conda shell.bash hook)"
conda activate wm
cd $PROJECT_DIR
CUDA_VISIBLE_DEVICES=0 python $PROJECT_DIR/Code2World/android_world/agents/wm_server.py \
    --model $WM_MODEL_PATH \
    --port $WM_PORT \
    --device cuda:0 &
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

# 额外清理：确保 Redis/端口 6379 没被占用
fuser -k 6379/tcp 2>/dev/null || true
sleep 1

# 启动 Ray Head (GPU 1 only, GPU 0 is for WM vLLM)
echo "Starting Ray HEAD on $(hostname) ($HEAD_NODE_IP)"
export CUDA_VISIBLE_DEVICES=1
nohup ray start --head --node-ip-address=$HEAD_NODE_IP --port=6379 \
    --num-gpus=1 --resources="{\"docker:$HEAD_NODE_IP\": 128}" \
    > /tmp/ray_head.log 2>&1 &
RAY_HEAD_PID=$!
echo "Ray head process PID: $RAY_HEAD_PID"

# 等几秒后检查进程是否还活着
sleep 3
if ! kill -0 $RAY_HEAD_PID 2>/dev/null; then
    echo "ERROR: ray start process exited immediately. Log:"
    cat /tmp/ray_head.log
    exit 1
fi

# 等待 GCS 就绪 (直接探测 6379 端口，Ray 初始化 GPU 可能需要几分钟)
echo "Waiting for Ray GCS to become available..."
GCS_READY=0
for i in $(seq 1 300); do
    if bash -c 'echo > /dev/tcp/127.0.0.1/6379' 2>/dev/null; then
        echo "GCS ready after ${i}s"
        GCS_READY=1
        break
    fi
    sleep 1
done
if [ "$GCS_READY" -ne 1 ]; then
    echo "ERROR: Ray GCS failed to start after 300s. Check /tmp/ray_head.log:"
    cat /tmp/ray_head.log
    exit 1
fi

# Give Ray internal services time to fully initialize
sleep 10

# Wait until Ray Python API is actually usable
echo "Waiting for Ray Python API to be ready..."
RAY_API_READY=0
for i in $(seq 1 120); do
    if python3 -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print('Ray API ready'); ray.shutdown()" 2>/dev/null; then
        RAY_API_READY=1
        break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        echo "Waiting for Ray API... ($i/120s)"
    fi
    sleep 1
done
if [ "$RAY_API_READY" -ne 1 ]; then
    echo "ERROR: Ray Python API not ready after 120s."
    echo "Ray head log:"
    cat /tmp/ray_head.log
    exit 1
fi

# 其他节点启动 Ray Worker + WM vLLM 服务
if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
    echo "Starting Ray WORKERS + WM servers on other nodes"

    # Launch worker node initialization via srun (foreground — blocks until workers finish setup)
    echo "Launching srun for worker nodes..."
    srun --nodes=$((SLURM_JOB_NUM_NODES - 1)) --ntasks-per-node=1 --ntasks=$((SLURM_JOB_NUM_NODES - 1)) --exclude=$(hostname) \
        bash -c '
            echo "[$(hostname)] Worker node srun started"
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
            export PROJECT_DIR='"$PROJECT_DIR"'
            export WM_MODEL_PATH='"$WM_MODEL_PATH"'

            # WM 服务 (GPU 0)
            WM_PORT=18888
            echo "[$(hostname)] Starting WM server on GPU:0"
            export CONDA_PREFIX=""
            eval "$(conda shell.bash hook)"
            conda activate wm
            export PYTHONPATH=$PYTHONPATH:$PROJECT_DIR/Code2World
            cd $PROJECT_DIR
            CUDA_VISIBLE_DEVICES=0 python $PROJECT_DIR/Code2World/android_world/agents/wm_server.py \
                --model $WM_MODEL_PATH \
                --port $WM_PORT \
                --device cuda:0 > /tmp/wm_server_worker.log 2>&1 &
            sleep 10
            echo "[$(hostname)] Waiting for WM server..."
            for i in $(seq 1 120); do
                if curl -sf http://127.0.0.1:${WM_PORT}/v1/models > /dev/null 2>&1; then
                    echo "[$(hostname)] WM server ready after ${i}s"
                    break
                fi
                sleep 1
            done

            # Ray Worker (GPU 1 only)
            echo "[$(hostname)] Starting Ray worker..."
            ray stop --force 2>/dev/null || true
            sleep 2
            WORKER_IP=$(hostname -I | awk "{print \$1}")
            echo "[$(hostname)] Worker IP: $WORKER_IP, connecting to head: $HEAD_NODE_IP:6379"
            export CUDA_VISIBLE_DEVICES=1
            ray start --address=$HEAD_NODE_IP:6379 --num-gpus=1 --resources="{\"docker:$WORKER_IP\": 128}" 2>&1 | tee /tmp/ray_worker_connect.log
            echo "[$(hostname)] Ray start finished, exit code: $?"
        '
    echo "All worker nodes initialized"
else
    echo "Single-node job, skipping worker node setup"
fi

# --- 2. 配置 ---
EXPERIMENT_NAME="wm_r1"
MODEL_PATH="/public/home/xlwang/genalyu/models/Qwen2.5-VL-3B-Instruct"
SYSTEM_PROMPT="You are a helpful assistant."

NUM_GPUS_PER_NODE=1         # 训练只用 GPU 1
NUM_NODES=$SLURM_JOB_NUM_NODES
ROLLOUT_N=$((3 * NUM_NODES))
NUM_ENVS=$((2 * ROLLOUT_N))  # env workers 必须等于 rollout_batch_size * rollout_n

TRAIN_FILES="/public/home/xlwang/genalyu/dataset/agentnet/agentnet_train_9k.jsonl"
VAL_FILES="/public/home/xlwang/genalyu/dataset/agentnet/agentnet_test_1k.jsonl"

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

# Fix GCC/linker mismatch for Triton JIT compilation
export LD_LIBRARY_PATH=/public/software/compiler/gnu/gcc-14.3.0/lib64:${LD_LIBRARY_PATH:-}
export CC=/public/software/compiler/gnu/gcc-14.3.0/bin/gcc
export AR=/public/software/compiler/gnu/gcc-14.3.0/bin/ar
export CUDA_HOST_COMPILER=$CC

# 限制线程数，避免 Ray worker 过多导致资源竞争
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# Ensure Ray Python API can auto-detect the cluster
export RAY_ADDRESS="127.0.0.1:6379"

python3 -m verl.trainer.main \
    trainer.project_name="wm-r1-agentnet" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="['console', 'wandb']" \
    trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.nnodes=$NUM_NODES \
    trainer.default_local_dir=/public/home/xlwang/genalyu/WM-R1/checkpoints/$SLURM_JOB_ID \
    trainer.save_freq=16 \
    trainer.save_limit=3 \
    trainer.val_before_train=True \
    trainer.val_freq=16 \
    trainer.total_episodes=15 \
    env.use_wm=True \
    env.use_local_wm=False \
    env.wm_http_url="${WM_HTTP_URL}" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.format_prompt="${SYSTEM_PROMPT}" \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
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
    worker.rollout.max_num_batched_tokens=3072 \
    algorithm.adv_estimator="wm_r1" \
    algorithm.disable_kl=True \
    env.num_envs=$NUM_ENVS \
    env.max_steps=15 \
    data.num_workers=4

# --- 5. 清理 ---
ray stop
