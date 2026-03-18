set -x

MODEL_PATH=UI-TARS-1.5-7B

SYSTEM_PROMPT="""You are helpful assistant."""

NUM_GPUS=8
NUM_ENVS=128
ROLLOUT_N=8

((ROLLOUT_BSZ = NUM_ENVS/ROLLOUT_N))


python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.format_prompt="${SYSTEM_PROMPT}" \
    data.train_files=evaluation_examples/test_success_uitars1.5_wo_impossible.json \
    data.val_files=evaluation_examples/test_success_uitars1.5_wo_impossible.json \
    data.max_prompt_length=64000 \
    data.max_response_length=8192 \
    data.max_pixels=2116800 \
    data.min_pixels=256 \
    data.rollout_batch_size=16 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.max_grad_norm=1.0 \
    worker.actor.optim.lr=1e-6 \
    worker.actor.ulysses_sequence_parallel_size=1 \
    worker.actor.padding_free=true \
    worker.actor.ppo_epochs=1 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.3 \
    worker.actor.global_batch_size=8 \
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
    env.num_envs=$NUM_ENVS \
    env.max_steps=15 \
    trainer.experiment_name=osworld_cot_7b_nokl_twonodes_onlinereplay \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=2 \
    trainer.save_freq=8 \
    trainer.save_limit=3 \
    trainer.val_before_train=True \
    trainer.val_freq=8 \
    trainer.total_episodes=15
