set -x

# 实验名称
EXPERIMENT_NAME="wm_r1_agentnet_experiment"

# 模型路径 (请根据实际情况修改)
MODEL_PATH="UI-TARS-1.5-7B"

# 系统提示词
SYSTEM_PROMPT="You are a helpful assistant."

# GPU 和 环境配置
NUM_GPUS=8
NUM_ENVS=16
ROLLOUT_N=8

# 数据集配置
# 1. 远程加载 (Hugging Face):
# TRAIN_FILES="xlangai/AgentNet@train"

# 2. 本地加载 (支持 .parquet, .jsonl 或包含这些文件的目录):
# 假设你已经将 AgentNet 下载到了 /path/to/local/AgentNet
TRAIN_FILES="/path/to/local/AgentNet"
VAL_FILES="/path/to/local/AgentNet"

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
