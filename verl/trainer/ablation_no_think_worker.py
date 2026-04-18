"""
Ablation worker: World Model for state prediction only.
The model generates ONE action per step (no call_wm deep thinking).
The WM predicts the next state, but the model cannot use <think> to imagine.
"""
import ray
import os
import torch
import numpy as np
import base64
import traceback
from io import BytesIO
from PIL import Image
from qwen_vl_utils import process_vision_info

from .gui_agent import (
    EnvWorker,
    WorldModelEnv,
    uitars_system_prompt,
    parse_action_to_structure_output,
    add_box_token,
    FINISH_WORD,
    WAIT_WORD,
    ENV_FAIL_WORD,
    CALL_USER,
)
import verl.utils.torch_functional as VF
from ..utils.tokenizer import get_processor, get_tokenizer
from ..models.transformers.qwen2_vl import get_rope_index


@ray.remote(num_cpus=1)
class AblationNoThinkWorker(EnvWorker):
    """
    Ablation: World model is used for state prediction only.
    The policy model generates ONE action per step.
    Deep thinking via call_wm() in <think> is disabled.
    """

    def __init__(self, worker_idx, max_steps=15, config=None):
        self.worker_idx = worker_idx
        self.step_timeout = 60
        self.config = config
        self.use_wm = True
        self.n_wm = 0
        self.n_wm_max = 0  # Force to 0: no WM deep thinking allowed

        self.tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        self.processor = get_processor(
            config.worker.actor.model.model_path,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        self.model = 'uitars'

        # Create WorldModelEnv for state prediction only
        checkpoint_path = getattr(config.trainer, "save_checkpoint_path", "checkpoints")
        html_save_dir = os.path.join(checkpoint_path, "wm_htmls", f"ablation_worker_{worker_idx}")

        self.env = WorldModelEnv(
            model_name_or_path=getattr(config.env, "wm_model_name", "internvl3-78b"),
            api_base=getattr(config.env, "wm_api_base", None),
            api_key=getattr(config.env, "wm_api_key", None),
            use_local_transformers=getattr(config.env, "use_local_wm", True),
            html_save_dir=html_save_dir
        )

        self.is_init = False
        self.is_done = False
        self.max_steps = max_steps

        self.action_parse_res_factor = 1000
        self.model_type = "qwen25vl"
        self.max_pixels = 16384*28*28
        self.min_pixels = 100*28*28

        self.instruction = None
        self.task_config = None
        self.step_counter = 0
        self.history_images = []
        self.history_messages = []

        self.reset_train_tensors()

    def step(self, prediction):
        """
        Same as parent EnvWorker.step but WITHOUT call_wm deep thinking support.
        """
        self.is_init = False

        # No deep thinking allowed in ablation - always treat as regular turn
        is_imaginary = False

        try:
            parsed_responses = parse_action_to_structure_output(
                prediction,
                self.action_parse_res_factor,
                1080,
                1920,
                self.model_type,
                self.max_pixels,
                self.min_pixels
            )

            actions = []
            action_desc_parts = []
            for parsed_response in parsed_responses:
                if "action_type" in parsed_response:
                    if parsed_response["action_type"] == FINISH_WORD:
                        actions = ['DONE']
                        break
                    elif parsed_response["action_type"] == WAIT_WORD:
                        actions = ['WAIT']
                        break
                    elif parsed_response["action_type"] == ENV_FAIL_WORD:
                        actions = ['FAIL']
                        break
                    elif parsed_response["action_type"] == CALL_USER:
                        actions = ['FAIL']
                        break

                    desc = f"{parsed_response.get('action_type', '')}({parsed_response.get('action_inputs', {})})"
                    action_desc_parts.append(desc)
                    actions.append(desc)

            action_desc = " | ".join(action_desc_parts)
            info = {"simulated": True}

            if actions[0] in ['DONE', 'FAIL', 'WAIT']:
                reward = 0.0
                step_done = (actions[0] != 'WAIT')
                obs_screenshot = self.history_images[-1]
            else:
                obs, reward, step_done, _ = self.env.step(action_desc)
                obs_screenshot = obs['screenshot']

                if isinstance(obs_screenshot, np.ndarray):
                    buffer = BytesIO()
                    Image.fromarray(obs_screenshot).save(buffer, format="JPEG")
                    obs_screenshot = buffer.getvalue()

            format_reward = 0.0
        except Exception as e:
            print('Ablation step error: ', prediction)
            print('Error traceback: ', traceback.format_exc())
            format_reward = -1.0
            actions = ['DONE']
            obs_screenshot = self.history_images[-1]
            reward = 0.0
            step_done = True
            info = {}

        if step_done:
            self.is_done = True

        self.step_counter += 1
        if self.step_counter == self.max_steps:
            self.is_done = True

        self.env.pause()

        self.history_images.append(obs_screenshot)
        self.history_messages.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": add_box_token(prediction)
            }]
        })

        if not self.is_done:
            if obs_screenshot is None:
                self.is_done = True
                self.process_message(self.history_messages[-1:])
                return {
                    'env_idx': self.worker_idx,
                    'obs_messages': None,
                    'is_done': self.is_done,
                    'format_reward': format_reward,
                    'n_wm': self.n_wm
                }

            image_base64 = base64.b64encode(BytesIO(obs_screenshot).getvalue()).decode('utf-8')

            self.history_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": ""
                    },
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_base64}",
                        "min_pixels": 3136,
                        "max_pixels": 2116800,
                    }
                ]
            })
            self.process_message(self.history_messages[-2:])
            return {
                'env_idx': self.worker_idx,
                'obs_messages': self.history_messages,
                'is_done': self.is_done,
                'format_reward': format_reward,
                'n_wm': self.n_wm
            }
        else:
            self.process_message(self.history_messages[-1:])
            return {
                'env_idx': self.worker_idx,
                'obs_messages': None,
                'is_done': self.is_done,
                'format_reward': format_reward,
                'n_wm': self.n_wm
            }
