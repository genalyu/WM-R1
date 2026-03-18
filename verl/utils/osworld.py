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

import time
import copy
import json
import random
import math
import os
from io import BytesIO
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from qwen_vl_utils import process_vision_info

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    return features

def collate_fn_dataproto(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def collate_fn_fake(features_list):
    features = []
    for f in features_list:
        features.extend(f)

    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class OSWorldTaskConfigDataset(Dataset):

    def __init__(
        self,
        data_path: str,
    ):
        self.data_path = data_path
        with open(data_path, "r") as f:
            task_configs = json.load(f)
        
        self.dataset = []
        for domain, task_config_list in task_configs.items():
            for task_config in task_config_list:
                self.dataset.append((domain, task_config))
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # if self.offline_data:
            # return self.getitem_offline(index)

        domain, task_id = self.dataset[index]

        baes_path = os.path.dirname(self.data_path)
        with open(os.path.join(baes_path, 'examples', domain, task_id + '.json'), "r") as f:
            task_config = json.load(f)
        
        task_config['domain'] = domain
        task_config['id'] = task_id
        task_config["task_id"] = task_id

        return task_config


class OSWorldDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        messages,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
        fast_rollout: bool = False,
    ):
        self.messages = messages
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.format_prompt = format_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.fast_rollout = fast_rollout

    def __len__(self):
        return len(self.messages)
    
    
    def __getitem__(self, index):
        message = self.messages[index]

        tokenizer = self.tokenizer
        processor = self.processor
        
        prompt = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
                message, return_video_kwargs=True)

        row_dict = dict()
        row_dict["multi_modal_data"] = {"image": image_inputs} # [PIL.Image, ...]

        if not self.fast_rollout: 
            # Multi-turn conversation tokenization
            model_inputs = processor(image_inputs, [prompt], add_special_tokens=False, return_tensors="pt")
        
            input_ids = model_inputs.pop('input_ids')[0]
            attention_mask = model_inputs.pop('attention_mask')[0]

            row_dict["multi_modal_inputs"] = dict(model_inputs)
            
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            # to make sure dataproto can be created
            input_ids = torch.zeros((0,), dtype=torch.int64) 
            attention_mask = torch.zeros((0,), dtype=torch.int64)
            position_ids = torch.zeros((3, 0), dtype=torch.int64)
            row_dict['multi_modal_inputs'] = dict()

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        row_dict["input_ids"] = input_ids
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["raw_prompt_ids"] = self.tokenizer.encode(prompt, add_special_tokens=False)
        return row_dict

import ray
@ray.remote(num_cpus=1)
class GRPODatasetProcessor:
    def __init__(self, procosser, tokenizer, max_prompt_length=65000, truncation="right"):
        self.processor = procosser
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

    def load_content(self, content):
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            return ''.join([self.load_content(c) for c in content])

        if isinstance(content, dict):
            if "text" in content:
                return content["text"]
            elif "image" in content:
                return "<|vision_start|><|image_pad|><|vision_end|>"
        
        raise ValueError(f"Unknown content type: {content}")

    def process(self, message, post_process=True):
        tokenizer = self.tokenizer
        processor = self.processor

        # get images from message
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            message, return_video_kwargs=True)
        
        if image_inputs is not None and len(image_inputs) >= 1:
            input_ids = []
            labels = []
            attention_mask = []

            prompts = []
            image_count = 0

            pixel_values = []
            image_grid_thw = []
            for turn_idx, msg in enumerate(message):
                role = msg['role']
                content = self.load_content(msg['content'])
                prompt = f'<|im_start|>{role}\n' + content + '<|im_end|>\n'
                prompts.append(prompt)

                cur_image_num = prompt.count("<|image_pad|>")                
                if cur_image_num > 0:
                    result = processor(image_inputs[image_count:image_count+cur_image_num], [prompt], add_special_tokens=False, return_tensors="pt")
                    image_count += cur_image_num
                else:
                    result = processor(None, [prompt], add_special_tokens=False, return_tensors="pt")
                
                cur_input_ids = result.pop('input_ids')[0]
                cur_attention_mask = result.pop('attention_mask')[0]
                if 'pixel_values' in result: # 10764, 1176
                    pixel_values.append(result["pixel_values"])
                if 'image_grid_thw' in result:
                    image_grid_thw.append(result["image_grid_thw"])
                

                input_ids.append(cur_input_ids)
                attention_mask.append(cur_attention_mask)
                if role in ["system", "user"]:
                    labels.append(torch.full_like(cur_input_ids, -100))
                else:
                    labels.append(cur_input_ids)

            input_ids = torch.cat(input_ids, dim=0)
            labels = torch.cat(labels, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)

            pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
            image_grid_thw = torch.cat(image_grid_thw, dim=0) if len(image_grid_thw) > 0 else None

            model_inputs = {
                'pixel_values': pixel_values,
                'image_grid_thw': image_grid_thw,
            }

            # pixel_values: (53820, 1176)
            # image_grid_thw: (5, 3)  [1, 78, 138]
            # prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            # result_old = processor(image_inputs, [prompt], add_special_tokens=False, return_tensors="pt")
            # prompts_cat = ''.join(prompts)
            # print(prompt == prompts_cat)
            # breakpoint()

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )
        else:
            input_ids = torch.zeros((0,), dtype=torch.int64)
            labels = torch.full((0,), IGNORE_INDEX, dtype=torch.int64)
            attention_mask = torch.zeros((0,), dtype=torch.int64)
            position_ids = torch.zeros((3, 0), dtype=torch.int64)
            model_inputs = dict()

        if post_process:
            input_ids, attention_mask, position_ids, labels = VF.postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
                labels=labels
            )
        row_dict = dict()
        row_dict["input_ids"] = input_ids
        row_dict["labels"] = labels
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["multi_modal_inputs"] = model_inputs
        return row_dict
        # return input_ids, labels, attention_mask, position_ids, model_inputs


class OSWorldGRPODataset(ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        messages,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: str = None,
        max_pixels: int = None,
        min_pixels: int = None,
    ):
        self.messages = messages
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.format_prompt = format_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels

    def __len__(self):
        return len(self.messages)
    
    
    def load_content(self, content):
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            return ''.join([self.load_content(c) for c in content])

        if isinstance(content, dict):
            if "text" in content:
                return content["text"]
            elif "image" in content:
                return "<|vision_start|><|image_pad|><|vision_end|>"
        
        raise ValueError(f"Unknown content type: {content}")


    def __getitem__(self, index):
        message = copy.deepcopy(self.messages[index])

        tokenizer = self.tokenizer
        processor = self.processor
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            message, return_video_kwargs=True)

        
        # prompt = processor.apply_chat_template(
        #         message,
        #         tokenize=False,
        #         add_generation_prompt=False,
        #     )
        
        if image_inputs is not None and len(image_inputs) >= 1:
            input_ids = []
            labels = []
            attention_mask = []

            prompts = []
            image_count = 0

            pixel_values = []
            image_grid_thw = []
            for turn_idx, msg in enumerate(message):
                role = msg['role']
                content = self.load_content(msg['content'])
                prompt = f'<|im_start|>{role}\n' + content + '<|im_end|>\n'
                prompts.append(prompt)

                cur_image_num = prompt.count("<|image_pad|>")                
                if cur_image_num > 0:
                    result = processor(image_inputs[image_count:image_count+cur_image_num], [prompt], add_special_tokens=False, return_tensors="pt")
                    image_count += cur_image_num
                else:
                    result = processor(None, [prompt], add_special_tokens=False, return_tensors="pt")
                
                cur_input_ids = result.pop('input_ids')[0]
                cur_attention_mask = result.pop('attention_mask')[0]
                if 'pixel_values' in result: # 10764, 1176
                    pixel_values.append(result["pixel_values"])
                if 'image_grid_thw' in result:
                    image_grid_thw.append(result["image_grid_thw"])
                

                input_ids.append(cur_input_ids)
                attention_mask.append(cur_attention_mask)
                if role in ["system", "user"]:
                    labels.append(torch.full_like(cur_input_ids, -100))
                else:
                    labels.append(cur_input_ids)

            input_ids = torch.cat(input_ids, dim=0)
            labels = torch.cat(labels, dim=0)
            attention_mask = torch.cat(attention_mask, dim=0)

            pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
            image_grid_thw = torch.cat(image_grid_thw, dim=0) if len(image_grid_thw) > 0 else None

            model_inputs = {
                'pixel_values': pixel_values,
                'image_grid_thw': image_grid_thw,
            }

            # pixel_values: (53820, 1176)
            # image_grid_thw: (5, 3)  [1, 78, 138]
            # prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            # result_old = processor(image_inputs, [prompt], add_special_tokens=False, return_tensors="pt")
            # prompts_cat = ''.join(prompts)
            # print(prompt == prompts_cat)
            # breakpoint()

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs["image_grid_thw"],
                attention_mask=attention_mask,
            )
        else:
            input_ids = torch.zeros((0,), dtype=torch.int64)
            labels = torch.full((0,), IGNORE_INDEX, dtype=torch.int64)
            attention_mask = torch.zeros((0,), dtype=torch.int64)
            position_ids = torch.zeros((3, 0), dtype=torch.int64)
            model_inputs = dict()


        input_ids, attention_mask, position_ids, labels = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
            labels=labels
        )
        row_dict = dict()
        row_dict["input_ids"] = input_ids
        row_dict["labels"] = labels
        row_dict["attention_mask"] = attention_mask
        row_dict["position_ids"] = position_ids
        row_dict["multi_modal_inputs"] = model_inputs
        return row_dict