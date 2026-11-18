#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR=/root/genalyu/WM-R1
cd "$PROJECT_DIR"
PYTHON="$PROJECT_DIR/.venv/bin/python"
export PATH="$PROJECT_DIR/.venv/bin:$PATH"
export PYTHONPATH="$PROJECT_DIR:$PROJECT_DIR/Code2World"
export WANDB_API_KEY="$(cat "$PROJECT_DIR/.secrets/wandb.key")"
export WANDB_ENTITY=genal-yu
export WANDB_PROJECT=wm-r1-android
export WANDB_MODE=online
export WANDB_BASE_URL=https://api.wandb.ai
export HTTPS_PROXY=http://127.0.0.1:17898
export HTTP_PROXY=http://127.0.0.1:17898
export NO_PROXY=localhost,127.0.0.1,"$(hostname -I | awk '{print $1}')"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export WM_FAST_RENDER=1 WM_SPEED_ONLY=1 WM_MAX_OUTPUT_TOKENS=64 WM_JUDGE_MAX_TOKENS=96
export VLLM_USE_V1=0
export TRITON_PTXAS_PATH="$PROJECT_DIR/.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas"
export WM_R1_ATTN_IMPLEMENTATION=sdpa
export VLLM_WORKER_MULTIPROCESS_METHOD=spawn
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
EXPERIMENT_NAME="speed-only-subset32-3b-seed1-$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$PROJECT_DIR/runs/$EXPERIMENT_NAME"
mkdir -p "$RUN_DIR"
export WANDB_DIR="$RUN_DIR"
export WM_HTTP_URL=http://127.0.0.1:18888
MODEL_PATH="$PROJECT_DIR/models/Qwen2.5-VL-3B-Instruct"
WM_MODEL_PATH="$PROJECT_DIR/models/Code2World"
TRAIN_FILES="$PROJECT_DIR/data/guir1_subset/train.jsonl"
VAL_FILES="$PROJECT_DIR/data/guir1_subset/val.jsonl"
NUM_GPUS_PER_NODE=1
ROLLOUT_N=4
NUM_ENVS=128
for f in "$TRAIN_FILES" "$VAL_FILES" "$MODEL_PATH/model.safetensors.index.json" "$WM_MODEL_PATH/model.safetensors.index.json"; do
 test -f "$f" || { echo "Missing asset: $f"; exit 1; }
done
"$PYTHON" - <<'CHECK'
import wandb
api=wandb.Api(timeout=20)
print('W&B entity:',api.default_entity)
CHECK
if [[ "${WM_EXTERNAL:-0}" == 1 ]]; then
 for attempt in $(seq 1 60); do
  if curl --noproxy '*' -sf "$WM_HTTP_URL/health" >/dev/null; then break; fi
  sleep 1
 done
 curl --noproxy '*' -fsS "$WM_HTTP_URL/health" >/dev/null
 for attempt in $(seq 1 180); do
  if curl --noproxy '*' -sf http://127.0.0.1:18890/health >/dev/null; then break; fi
  sleep 2
 done
 curl --noproxy '*' -fsS http://127.0.0.1:18890/health >/dev/null
else
WM_BATCH_SIZE="${WM_BATCH_SIZE:-4}"
if [[ "$WM_BATCH_SIZE" -gt 1 ]]; then
 WM_ARGS=(Code2World/android_world/agents/wm_batch_server.py --model "$WM_MODEL_PATH" --device cuda:0 --port 18888 --max-batch-size "$WM_BATCH_SIZE" --batch-wait-ms 30)
else
 WM_ARGS=(Code2World/android_world/agents/wm_server.py --model "$WM_MODEL_PATH" --device cuda:0 --port 18888 --gpu-memory-utilization 0.85)
fi
CUDA_VISIBLE_DEVICES=0 "$PYTHON" -u "${WM_ARGS[@]}" > "$RUN_DIR/wm-server.log" 2>&1 &
WM_PID=$!
trap 'kill "$WM_PID" 2>/dev/null || true' EXIT
ready=0
for i in $(seq 1 180); do
 if curl --noproxy '*' -sf "$WM_HTTP_URL/health" >/dev/null; then ready=1; break; fi
 kill -0 "$WM_PID" 2>/dev/null || { tail -50 "$RUN_DIR/wm-server.log"; exit 1; }
 sleep 2
done
[[ "$ready" == 1 ]] || { echo 'WM server not ready'; exit 1; }
fi
export CUDA_VISIBLE_DEVICES=1
export RAY_ADDRESS=localhost:6379
printf '%s\n' "$RUN_DIR" > "$PROJECT_DIR/runs/latest-run.txt"
"$PYTHON" -m verl.trainer.main \
    trainer.project_name="wm-r1-android" trainer.save_checkpoint_path="${RUN_DIR}/checkpoints" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="['console', 'wandb']" \
    trainer.n_gpus_per_node=$NUM_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=4 \
    trainer.save_limit=3 \
    trainer.val_before_train=False \
    trainer.val_freq=999999 \
    trainer.total_episodes=3 trainer.max_steps=3 \
    \
    env.platform="android" \
    env.use_wm=True \
    env.use_local_wm=False \
    env.wm_http_url="${WM_HTTP_URL}" \
    env.num_envs=$NUM_ENVS \
    env.max_steps=5 \
    env.n_wm_max=0 \
    \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.format_prompt="You are a helpful assistant." \
    data.max_prompt_length=256 \
    data.max_response_length=48 \
    data.max_pixels=3136 \
    data.min_pixels=784 \
    data.rollout_batch_size=32 \
    data.num_workers=2 \
    data.max_trajectory_steps=1 \
    \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.enable_gradient_checkpointing=False \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.ppo_epochs=1 \
    worker.actor.global_batch_size=32 \
    worker.actor.micro_batch_size_per_device_for_update=16 \
    worker.actor.micro_batch_size_per_device_for_experience=64 \
    worker.actor.use_torch_compile=False \
    \
    worker.rollout.gpu_memory_utilization=0.30 \
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
    reward.gamma=0.1 \
    data.val_batch_size=2

