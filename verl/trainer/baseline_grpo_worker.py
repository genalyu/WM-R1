"""
Baseline GRPO worker: No World Model at all.
The model generates the entire trajectory in one response.
"""
import ray
import torch
from qwen_vl_utils import process_vision_info
import base64
from io import BytesIO
import numpy as np
from PIL import Image

from .gui_agent import (
    EnvWorker,
    uitars_system_prompt,
    parse_action_to_structure_output,
    add_box_token,
    get_rope_index,
)
import verl.utils.torch_functional as VF
from ..utils.tokenizer import get_processor, get_tokenizer


@ray.remote(num_cpus=1)
class BaselineEnvWorker(EnvWorker):
    """
    Baseline worker: no world model simulation.
    The policy model generates the complete trajectory of actions in a single response.
    Only one step is taken — the model outputs all actions at once.
    """

    def __init__(self, worker_idx, max_steps=15, config=None):
        # Call parent __init__ but skip WorldModelEnv creation
        self.worker_idx = worker_idx
        self.step_timeout = 60
        self.config = config
        self.use_wm = False  # Always False for baseline
        self.n_wm = 0
        self.n_wm_max = 0

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
        self.env = None  # No environment for baseline

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

    def reset(self, task_config):
        self.instruction = task_config.get("instruction", None)
        self.task_config = task_config
        self.step_counter = 0
        self.is_done = False

        self.reset_train_tensors()

        init_screenshot = task_config.get("init_screenshot", None)
        if init_screenshot is None:
            init_screenshot = np.zeros((2400, 1080, 3), dtype=np.uint8)

        if isinstance(init_screenshot, bytes):
            init_screenshot = np.array(Image.open(BytesIO(init_screenshot)))

        image_base64 = base64.b64encode(BytesIO(init_screenshot).getvalue()).decode("utf-8")

        init_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Your are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt.format(instruction=self.instruction)
                        + "\n\nPlease generate ALL actions needed to complete this task in a single response. "
                        + "Output each action on a separate line with 'Action:' prefix."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_base64}",
                        "min_pixels": 3136,
                        "max_pixels": 2116800,
                    }
                ]
            }
        ]

        self.history_images = [init_screenshot]
        self.history_messages = init_messages
        self.process_message(init_messages)

        return {
            'env_idx': self.worker_idx,
            'obs_messages': self.history_messages,
            'is_done': self.is_done,
            'format_reward': 0.0
        }

    def step(self, prediction):
        """
        Baseline step: the model generates the complete trajectory.
        We parse all actions from the response and mark as done immediately.
        """
        self.is_init = False
        self.is_done = True  # Single-step baseline: all actions generated at once
        self.step_counter = 1

        format_reward = 0.0
        actions = []
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
            for parsed_response in parsed_responses:
                action_type = parsed_response.get("action_type", "")
                actions.append(action_type)
            format_reward = 0.5  # Format is correct if parsing succeeds
        except Exception as e:
            print(f'Baseline step error: {e}')
            format_reward = -1.0
            actions = ['DONE']

        # Record the response in history
        self.history_messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": add_box_token(prediction)}]
        })
        self.process_message(self.history_messages[-1:])

        return {
            'env_idx': self.worker_idx,
            'obs_messages': None,  # No next observation for baseline
            'is_done': self.is_done,
            'format_reward': format_reward,
            'n_wm': self.n_wm
        }
