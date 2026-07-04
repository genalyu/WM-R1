#!/bin/bash
# ============================================================
# WM-R1 单机部署 + 运行指南（Android 移动端实验）
# 目标：单张 H200 (141GB)，从零到跑出主实验
# ============================================================
#
# 使用方法：
#   1. 先通读本脚本，按需修改 "用户配置区"
#   2. 分步执行（不要一次全跑，每步验证后再继续）
#
# ============================================================


# ================================================================
# STEP 0: 用户配置（按需修改）
# ================================================================
PROJECT_DIR="$HOME/WM-R1"           # 项目目录
CONDA_ENV="wmr1"                    # conda 环境名
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"   # Agent 策略模型
WM_MODEL_PATH="GD-ML/Code2World"             # World Model (Code2World 微调权重，基于 Qwen3-VL-8B)
DATA_DIR="$PROJECT_DIR/data"


# ================================================================
# STEP 1: 基础环境
# ================================================================
echo "=== Step 1: 基础环境 ==="

# 1a. Conda 环境
conda create -n "$CONDA_ENV" python=3.10 -y
conda activate "$CONDA_ENV"

# 1b. PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 1c. 核心依赖
pip install vllm>=0.6.0
pip install flash-attn --no-build-isolation
pip install ray>=2.10

# 1d. 项目依赖
pip install omegaconf psutil Pillow datasets liger_kernel \
    mlflow wandb qwen_vl_utils protobuf accelerate tensordict torchdata \
    codetiming mathruler swanlab

# 1e. Clone 项目
if [ ! -d "$PROJECT_DIR" ]; then
    git clone --recurse-submodules git@github.com:genalyu/WM-R1.git "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

# 1f. 安装项目本身
pip install -e .

# 1g. 安装 Code2World 的 Playwright 依赖（WM 渲染用）
cd Code2World
pip install -r requirements.txt
playwright install chromium
cd "$PROJECT_DIR"

# 验证
python -c "import torch; print(f'CUDA: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_mem/1e9:.0f}GB')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import ray; print(f'Ray: {ray.__version__}')"
echo "Step 1 完成 ✓"


# ================================================================
# STEP 2: 下载模型
# ================================================================
echo "=== Step 2: 下载模型 ==="

# Agent 策略模型 (~6GB)
huggingface-cli download "$MODEL_PATH" --local-dir "$PROJECT_DIR/models/Qwen2.5-VL-3B-Instruct"

# World Model (~16GB) — Code2World 微调权重（基于 Qwen3-VL-8B，在 AndroidCode 80K 对上 SFT + RL）
huggingface-cli download "$WM_MODEL_PATH" --local-dir "$PROJECT_DIR/models/Code2World"

# 验证
ls -lh "$PROJECT_DIR/models/Qwen2.5-VL-3B-Instruct/config.json"
ls -lh "$PROJECT_DIR/models/Code2World/config.json"
echo "Step 2 完成 ✓"


# ================================================================
# STEP 3: 准备数据
# ================================================================
echo "=== Step 3: 准备数据 ==="

mkdir -p "$DATA_DIR"

# 3a. AndroidControl
python scripts/convert_android_control.py \
    --source hf \
    --output "$DATA_DIR/android_control/"

# 3b. GUI-Odyssey
python scripts/convert_gui_odyssey.py \
    --source hf \
    --output "$DATA_DIR/gui_odyssey/"

# 3c. 合并两个数据集
python -c "
import json, random

records = []
for path in ['$DATA_DIR/android_control/train.jsonl', '$DATA_DIR/gui_odyssey/train.jsonl']:
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

random.seed(42)
random.shuffle(records)

# 划分 train/test
test_size = min(200, len(records) // 10)
train = records[test_size:]
test = records[:test_size]

with open('$DATA_DIR/android_train.jsonl', 'w') as f:
    for r in train:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

with open('$DATA_DIR/android_test.jsonl', 'w') as f:
    for r in test:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f'合并完成: train={len(train)}, test={len(test)}')
"

# 验证
head -1 "$DATA_DIR/android_train.jsonl" | python -m json.tool
echo "Step 3 完成 ✓"


# ================================================================
# STEP 4: 启动 World Model 服务
# ================================================================
echo "=== Step 4: 启动 WM 服务 ==="

# 开一个单独的 terminal / tmux 窗口运行:
#
#   cd $PROJECT_DIR
#   conda activate $CONDA_ENV
#   CUDA_VISIBLE_DEVICES=0 python Code2World/android_world/agents/wm_server.py \
#       --model $PROJECT_DIR/models/Code2World \
#       --port 18888 \
#       --device cuda:0
#
# 然后等它输出 "WM server ready" 后继续

# 验证 WM 服务 (在另一个 terminal):
#   curl -sf http://127.0.0.1:18888/v1/models | python -m json.tool
# 应该返回模型信息

echo "WM 服务需要在另一个 terminal 启动，见上方注释"
echo "Step 4: 手动启动 WM 服务后继续"


# ================================================================
# STEP 5: 启动训练
# ================================================================
echo "=== Step 5: 启动训练 ==="

# 清理 Ray 残留
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray ~/.config/ray/
pkill -9 -f raylet 2>/dev/null || true
pkill -9 -f gcs_server 2>/dev/null || true
sleep 2

# 环境变量
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="wm-r1-android"
export WANDB_MODE="offline"
export TORCH_COMPILE_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONPATH="$PROJECT_DIR/Code2World"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# 启动 Ray (单节点单卡)
ray start --head --port=6379 --num-gpus=1
sleep 5

# 验证 Ray
ray status

# 训练配置
TRAIN_FILES="$DATA_DIR/android_train.jsonl"
VAL_FILES="$DATA_DIR/android_test.jsonl"

# 启动主实验
cd "$PROJECT_DIR"
python3 -m verl.trainer.main \
    trainer.project_name="wm-r1-android" \
    trainer.experiment_name="wm_r1_android_main" \
    trainer.logger="['console', 'wandb']" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=8 \
    trainer.save_limit=3 \
    trainer.val_before_train=False \
    trainer.val_freq=16 \
    trainer.total_episodes=15 \
    \
    env.platform="android" \
    env.use_wm=True \
    env.use_local_wm=False \
    env.wm_http_url="http://127.0.0.1:18888" \
    env.num_envs=4 \
    env.max_steps=15 \
    env.n_wm_max=5 \
    \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.format_prompt="You are a helpful assistant." \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.max_pixels=512000 \
    data.min_pixels=64000 \
    data.rollout_batch_size=2 \
    data.num_workers=2 \
    data.max_trajectory_steps=3 \
    \
    worker.actor.model.model_path="$PROJECT_DIR/models/Qwen2.5-VL-3B-Instruct" \
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
    worker.rollout.n=2 \
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
    reward.gamma=0.1 \
    data.val_batch_size=2

# 清理
ray stop
echo "训练完成"
