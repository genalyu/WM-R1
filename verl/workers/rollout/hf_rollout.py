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
A simple Hugging Face Transformers-based rollout for fallback.
"""

import torch
from typing import List, Union
import numpy as np
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, PreTrainedModel

from ...protocol import DataProto
from ...utils import torch_functional as VF
from .base import BaseRollout
from .config import RolloutConfig


class HFRollout(BaseRollout):
    def __init__(self, model: PreTrainedModel, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        # Basic HF generation
        # Note: This is a simplified version and might need more parameters from config
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.config.response_length,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            pad_token_id=self.pad_token_id,
            eos_token_id=eos_token_id,
        )

        # Extract only the newly generated tokens
        response_ids = outputs[:, input_ids.shape[-1]:]
        
        # Pad response_ids to max_length if necessary
        response_ids = VF.pad_2d_list_to_length(
            response_ids.tolist(), self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)

        # Replace image/video token IDs in generated responses with pad tokens.
        image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        for tok_id in [image_token_id, video_token_id]:
            if tok_id is not None:
                response_ids.masked_fill_(response_ids == tok_id, self.pad_token_id)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        
        # Update attention mask and position ids for the whole sequence
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)
        
        # Simple position id update (might need MRope handling for Qwen2-VL)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3: # qwen2vl mrope
             delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch=prompts.non_tensor_batch)
