import numpy as np
import math
import json
import random
from collections import defaultdict

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto, collate_fn

class ReplayBuffer():

    def __init__(self, json_path, buffer_size):
        self.buffer_size = buffer_size

        self.pos_dataset = defaultdict(list)

        # if json_path is not None:
        #     with open(json_path, 'r') as f:
        #         replay_data = json.load(f)
        
        #     for data in replay_data:
        #         # task_id, history_images, history_messages, eval_result
        #         task_id = data['task_id']
        #         eval_result = data['eval_result']
        #         if eval_result > 0.1:
        #             self.pos_dataset[task_id].append(data)
            
    
    def update_replay_buffer(self, task_config, batch_item, eval_result):
        task_id = task_config["task_id"]
        if eval_result > 0.1:
            task_replay_buffer = self.pos_dataset[task_id]
        else:
            return 

        task_replay_buffer.append(batch_item)

        if len(task_replay_buffer) > self.buffer_size:
            task_replay_buffer.pop(0)

    def update_replay_buffer_batch(self, task_configs, batch):
        eval_results = batch.batch['eval_results'].tolist()

        for task_config, batch_item, eval_result in zip(task_configs, batch, eval_results):
            self.update_replay_buffer(task_config, batch_item, eval_result)

    def get_pos(self, task_id, num_samples=1):
        if task_id not in self.pos_dataset:
            return DataProto()
        else:
            datalist = random.choices(self.pos_dataset[task_id], k=num_samples)
            return collate_fn(datalist)
    