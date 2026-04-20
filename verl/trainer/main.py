# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import json

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import CustomRewardManager
from .config import PPOConfig
from .ray_trainer import RayPPOTrainer, ResourcePoolManager, Role


@ray.remote(num_cpus=1)
class Runner:
    """A runner for RL training."""

    def run(self, config: PPOConfig):
        # print config
        config.deep_post_init()
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        reward_fn = CustomRewardManager(tokenizer=tokenizer, config=config.worker.reward)
        val_reward_fn = CustomRewardManager(tokenizer=tokenizer, config=config.worker.reward)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(PPOConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    ppo_config = OmegaConf.merge(default_config, cli_args)
    ppo_config = OmegaConf.to_object(ppo_config)

    if not ray.is_initialized():
        import os
        import subprocess
        import time

        ray_address = os.environ.get("RAY_ADDRESS", "")

        def _check_gcs_alive(addr: str) -> bool:
            """Quick health check: try connecting to Ray GCS."""
            try:
                result = subprocess.run(
                    ["python3", "-c",
                     f"import ray; ray.init(address='{addr}', ignore_reinit_error=True); ray.shutdown()"],
                    capture_output=True, timeout=300,
                )
                return result.returncode == 0
            except subprocess.TimeoutExpired:
                return False

        # If an existing cluster was specified, verify GCS is actually responsive
        if ray_address:
            print(f"Checking Ray cluster health at {ray_address}...")
            if not _check_gcs_alive(ray_address):
                print("WARNING: Existing Ray cluster GCS is not responsive, restarting...")
                subprocess.run(["ray", "stop", "--force"], capture_output=True)
                time.sleep(2)
                subprocess.run(["rm", "-rf", "/tmp/ray"], capture_output=True)
                ray_address = ""  # Force fresh start below

        if ray_address:
            # Existing cluster is healthy — connect to it
            print(f"Connecting to existing Ray cluster at {ray_address}...")
            ray.init(address="auto", ignore_reinit_error=True, runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "false",
                "NCCL_DEBUG": "WARN",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "OMP_NUM_THREADS": "4",
            }})
        else:
            # No existing cluster or it died — start fresh
            print("Starting a new Ray cluster...")
            subprocess.run(["ray", "start", "--head", "--port", "6379"], capture_output=True)
            for i in range(60):
                if _check_gcs_alive("localhost:6379"):
                    print(f"Ray cluster ready after {i}s")
                    break
                time.sleep(1)
            ray.init(address="localhost:6379", ignore_reinit_error=True, runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "false",
                "NCCL_DEBUG": "WARN",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "OMP_NUM_THREADS": "4",
            }})
    
    print(ray.cluster_resources().keys())

    runner = Runner.remote()
    ray.get(runner.run.remote(ppo_config))


if __name__ == "__main__":
    main()
