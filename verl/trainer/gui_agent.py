import ast
import re
import json
import math
from PIL import Image
import ray
import os
from io import BytesIO
import base64
import datetime
import traceback
import torch
import numpy as np
import sys
from qwen_vl_utils import process_vision_info

# Add Code2World to path to import wm_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Code2World'))
from android_world.agents.wm_utils import MLLMInferencer, render_aligned_png, _post_figure_action
from android_world.agents.qwen_model import QwenModel

# DesktopEnv is no longer used
# from desktop_env.desktop_env import DesktopEnv


class WorldModelEnv:
    def __init__(self, model_name_or_path="internvl3-78b", api_base=None, api_key=None, use_local_transformers=True, html_save_dir="wm_htmls"):
        if use_local_transformers:
            print(f"Loading local Transformers model: {model_name_or_path}")
            self.inferencer = QwenModel(model_name=model_name_or_path)
        else:
            self.inferencer = MLLMInferencer(model_name=model_name_or_path, api_base=api_base, api_key=api_key)
        
        self.current_screenshot = None
        self.task_config = None
        self.html_save_dir = html_save_dir
        os.makedirs(self.html_save_dir, exist_ok=True)
        self.step_idx = 0

    def reset(self, task_config):
        self.task_config = task_config
        self.instruction = task_config.get("instruction", "")
        self.current_screenshot = task_config.get("init_screenshot", None)
        if self.current_screenshot is None:
            # Fallback to a blank image if no init_screenshot
            self.current_screenshot = np.zeros((2400, 1080, 3), dtype=np.uint8)
        
        if isinstance(self.current_screenshot, bytes):
            self.current_screenshot = np.array(Image.open(BytesIO(self.current_screenshot)))
            
        self.step_idx = 0
        return {"screenshot": self.current_screenshot}

    def step(self, action_text, pause=0.5):
        self.step_idx += 1
        
        # Predict next state using World Model
        if isinstance(self.inferencer, QwenModel):
            # QwenModel.post handles saving HTML and rendering to PNG internally
            # We use action_text for both goal and description as a fallback
            next_state_array = self.inferencer.post(
                goal=getattr(self, "instruction", action_text),
                description=action_text,
                image=Image.fromarray(self.current_screenshot),
                action=action_text,
                html_save_path=self.html_save_dir,
                idx=self.step_idx
            )
        else:
            html_path = os.path.join(self.html_save_dir, f"step_{self.step_idx}") # _post_figure_action appends .html
            next_state_array = _post_figure_action(
                inferencer=self.inferencer,
                image_array=self.current_screenshot,
                action=action_text,
                html_save_path=html_path,
                idx=self.step_idx
            )
        
        if next_state_array is not None:
            self.current_screenshot = next_state_array
        
        # In World Model mode, we don't have real rewards or step_done
        # unless we implement a separate evaluator.
        return {"screenshot": self.current_screenshot}, 0.0, False, {}

    def pause(self):
        pass

    def unpause(self):
        pass

    def evaluate(self, instruction=None):
        """
        Use LLM as a judge to evaluate if the task is completed based on the final screenshot.
        """
        if instruction is None:
            if self.task_config:
                instruction = self.task_config.get("instruction", "")
            else:
                return 0.0

        judge_prompt = f"""You are an objective judge evaluating whether a GUI agent has successfully completed a task.
The user's original instruction is: "{instruction}"

Analyze the provided final screenshot of the computer screen. 
Determine if the goal described in the instruction has been achieved.

Output your judgment in the following JSON format:
{{
  "reasoning": "A brief explanation of why you think the task is completed or not.",
  "success": true/false
}}

Output ONLY the JSON.
"""
        try:
            # The current_screenshot is already updated at each step
            response_text, _, _ = self.inferencer.predict_mm(judge_prompt, [self.current_screenshot])
            
            # Extract JSON from response
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                is_success = data.get("success", False)
                print(f"Judge reasoning: {data.get('reasoning', 'No reasoning provided')}")
                return 1.0 if is_success else 0.0
            else:
                print(f"Failed to parse judge response: {response_text}")
                return 0.0
        except Exception as e:
            print(f"Error during LLM evaluation: {e}")
            return 0.0




uitars_system_prompt = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Use English in `Thought` and `Action` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

FINISH_WORD = "finished"
WAIT_WORD = "wait"
ENV_FAIL_WORD = "error_env"
CALL_USER = "call_user"

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

# 定义一个函数来解析每个 action
def parse_action(action_str):
    try:
        # 解析字符串为 AST 节点
        node = ast.parse(action_str, mode='eval')

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 获取函数名
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # 获取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值，这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {
            'function': func_name,
            'args': kwargs
        }

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None

def escape_single_quotes(text):
    # 匹配未转义的单引号（不匹配 \\'）
    pattern = r"(?<!\\)'"
    return re.sub(pattern, r"\\'", text)


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def linear_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    if width * height > max_pixels:
        """
        如果图片超过/低于像素限制，则计算一个缩放因子resize_factor，使图片的像素数缩小到等于或小于max_pixels。这个缩放因子是通过开平方根计算的，确保纵横比保持不变,这样原始的相对坐标可以不经转换直接复用
        """
        resize_factor = math.sqrt(max_pixels / (width * height))
        width, height = int(width * resize_factor), int(height * resize_factor)
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        width, height = math.ceil(width * resize_factor), math.ceil(height * resize_factor)

    return height, width 

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar



def parse_action_to_structure_output(text, factor, origin_resized_height, origin_resized_width, model_type, max_pixels=16384*28*28, min_pixels=100*28*28):
    text = text.strip()
    if model_type == "qwen25vl":
        smart_resize_height, smart_resize_width = smart_resize(origin_resized_height, origin_resized_width, factor=IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels)

    # 正则表达式匹配 Action 字符串
    if text.startswith("Thought:"):
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    elif text.startswith("Reflection:"):
        thought_pattern = r"Reflection: (.+?)Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Reflection: "
    elif text.startswith("Action_Summary:"):
        thought_pattern = r"Action_Summary: (.+?)(?=\s*Action:|$)"
        thought_hint = "Action_Summary: "
    else:
        thought_pattern = r"Thought: (.+?)(?=\s*Action:|$)"
        thought_hint = "Thought: "
    reflection, thought = None, None
    thought_match = re.search(thought_pattern, text, re.DOTALL)
    if thought_match:
        if len(thought_match.groups()) == 1:
            thought = thought_match.group(1).strip()
        elif len(thought_match.groups()) == 2:
            thought = thought_match.group(2).strip()
            reflection = thought_match.group(1).strip()
    assert "Action:" in text
    action_str = text.split("Action:")[-1]

    tmp_all_action = action_str.split("\n\n")
    all_action = []
    for action_str in tmp_all_action:
        if "type(content" in action_str:
            # 正则表达式匹配 content 中的字符串并转义单引号
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content

            # 使用正则表达式进行替换
            pattern = r"type\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "type(content='" + action_str + "')"
        elif "finished(content" in action_str:
            def escape_quotes(match):
                content = match.group(1)  # 获取 content 的值
                return content
            
            # 使用正则表达式进行替换
            pattern = r"finished\(content='(.*?)'\)"  # 匹配 type(content='...')
            content = re.sub(pattern, escape_quotes, action_str)

            # 处理字符串
            action_str = escape_single_quotes(content)
            action_str = "finished(content='" + action_str + "')"
            
        all_action.append(action_str)

    parsed_actions = [parse_action(action.replace("\n","\\n").lstrip()) for action in all_action]
    actions = []
    for action_instance, raw_str in zip(parsed_actions, all_action):
        if action_instance == None:
            print(f"Action can't parse: {raw_str}")
            raise ValueError(f"Action can't parse: {raw_str}") 
        action_type = action_instance["function"]
        params = action_instance["args"]

        # import pdb; pdb.set_trace()
        action_inputs = {}
        for param_name, param in params.items():
            if param == "": continue
            param = param.lstrip()  # 去掉引号和多余的空格
            # 处理start_box或者end_box参数格式 '<bbox>x1 y1 x2 y2</bbox>'
            action_inputs[param_name.strip()] = param
            
            if "start_box" in param_name or "end_box" in param_name:
                ori_box = param
                # Remove parentheses and split the string by commas
                numbers = ori_box.replace("(", "").replace(")", "").split(",")

                # Convert to float and scale by 1000
                # Qwen2.5vl output absolute coordinates, qwen2vl output relative coordinates
                if model_type == "qwen25vl":
                    float_numbers = []
                    for num_idx, num in enumerate(numbers):
                        num = float(num)
                        if (num_idx + 1) % 2 == 0:
                            float_numbers.append(float(num/smart_resize_height))
                        else:
                            float_numbers.append(float(num/smart_resize_width))
                else:
                    float_numbers = [float(num) / factor for num in numbers]

                if len(float_numbers) == 2:
                    float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                action_inputs[param_name.strip()] = str(float_numbers)

        # import pdb; pdb.set_trace()
        actions.append({
            "reflection": reflection,
            "thought": thought,
            "action_type": action_type,
            "action_inputs": action_inputs,
            "text": text
        })
    return actions

def parsing_response_to_pyautogui_code(responses, image_height: int, image_width:int, input_swap:bool=True) -> str:
    '''
    将M模型的输出解析为OSWorld中的action，生成pyautogui代码字符串
    参数:
        response: 包含模型输出的字典，结构类似于：
        {
            "action_type": "hotkey",
            "action_inputs": {
                "hotkey": "v ctrl",
                "start_box": None,
                "end_box": None
            }
        }
    返回:
        生成的pyautogui代码字符串
    '''

    pyautogui_code = f"import pyautogui\nimport time\n"
    if isinstance(responses, dict):
        responses = [responses]
    for response_id, response in enumerate(responses):
        if "observation" in response:
            observation = response["observation"]
        else:
            observation = ""

        if "thought" in response:
            thought = response["thought"]
        else:
            thought = ""
        
        if response_id == 0:
            pyautogui_code += f"'''\nObservation:\n{observation}\n\nThought:\n{thought}\n'''\n"
        else:
            pyautogui_code += f"\ntime.sleep(1)\n"

        action_dict = response
        action_type = action_dict.get("action_type")
        action_inputs = action_dict.get("action_inputs", {})
        
        if action_type == "hotkey":
            # Parsing hotkey action
            if "key" in action_inputs:
                hotkey = action_inputs.get("key", "")
            else:
                hotkey = action_inputs.get("hotkey", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"
            
            elif hotkey == "arrowup":
                hotkey = "up"
            
            elif hotkey == "arrowdown":
                hotkey = "down"

            if hotkey:
                # Handle other hotkeys
                keys = hotkey.split()  # Split the keys by space
                convert_keys = []
                for key in keys:
                    if key == "space":
                        key = ' '
                    convert_keys.append(key)
                pyautogui_code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"
        
        elif action_type == "press":
            # Parsing press action
            if "key" in action_inputs:
                key_to_press = action_inputs.get("key", "")
            else:
                key_to_press = action_inputs.get("press", "")

            if hotkey == "arrowleft":
                hotkey = "left"

            elif hotkey == "arrowright":
                hotkey = "right"
            
            elif hotkey == "arrowup":
                hotkey = "up"
            
            elif hotkey == "arrowdown":
                hotkey = "down"
            
            elif hotkey == "space":
                hotkey = " "
                
            if key_to_press:
                # Simulate pressing a single key
                pyautogui_code += f"\npyautogui.press({repr(key_to_press)})"
            
        elif action_type == "keyup":
            key_to_up = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyUp({repr(key_to_up)})"
        
        elif action_type == "keydown":
            key_to_down = action_inputs.get("key", "")
            pyautogui_code += f"\npyautogui.keyDown({repr(key_to_down)})"

        elif action_type == "type":
            # Parsing typing action using clipboard
            content = action_inputs.get("content", "")
            content = escape_single_quotes(content)
            stripped_content = content
            if content.endswith("\n") or content.endswith("\\n"):
                stripped_content = stripped_content.rstrip("\\n").rstrip("\n")
            if content:
                if input_swap:
                    pyautogui_code += f"\nimport pyperclip"
                    pyautogui_code += f"\npyperclip.copy('{stripped_content}')"
                    pyautogui_code += f"\npyautogui.hotkey('ctrl', 'v')"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"
                else:
                    pyautogui_code += f"\npyautogui.write('{stripped_content}', interval=0.1)"
                    pyautogui_code += f"\ntime.sleep(0.5)\n"
                    if content.endswith("\n") or content.endswith("\\n"):
                        pyautogui_code += f"\npyautogui.press('enter')"

        
        elif action_type in ["drag", "select"]:
            # Parsing drag or select action based on start and end_boxes
            start_box = action_inputs.get("start_box")
            end_box = action_inputs.get("end_box")
            if start_box and end_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                sx = round(float((x1 + x2) / 2) * image_width, 3)
                sy = round(float((y1 + y2) / 2) * image_height, 3)
                x1, y1, x2, y2 = eval(end_box)  # Assuming box is in [x1, y1, x2, y2]
                ex = round(float((x1 + x2) / 2) * image_width, 3)
                ey = round(float((y1 + y2) / 2) * image_height, 3)
                pyautogui_code += (
                    f"\npyautogui.moveTo({sx}, {sy})\n"
                    f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)\n"
                )

        elif action_type == "scroll":
            # Parsing scroll action
            start_box = action_inputs.get("start_box")
            if start_box:
                x1, y1, x2, y2 = eval(start_box)  # Assuming box is in [x1, y1, x2, y2]
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                
                # # 先点对应区域，再滚动
                # pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
            else:
                x = None
                y = None
            direction = action_inputs.get("direction", "")
            
            if x == None:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5)"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5)"
            else:
                if "up" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(5, x={x}, y={y})"
                elif "down" in direction.lower():
                    pyautogui_code += f"\npyautogui.scroll(-5, x={x}, y={y})"

        elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
            # Parsing mouse click actions
            start_box = action_inputs.get("start_box")
            start_box = str(start_box)
            if start_box:
                start_box = eval(start_box)
                if len(start_box) == 4:
                    x1, y1, x2, y2 = start_box  # Assuming box is in [x1, y1, x2, y2]
                elif len(start_box) == 2:
                    x1, y1 = start_box
                    x2 = x1
                    y2 = y1
                x = round(float((x1 + x2) / 2) * image_width, 3)
                y = round(float((y1 + y2) / 2) * image_height, 3)
                if action_type == "left_single" or action_type == "click":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='left')"
                elif action_type == "left_double":
                    pyautogui_code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
                elif action_type == "right_single":
                    pyautogui_code += f"\npyautogui.click({x}, {y}, button='right')"
                elif action_type == "hover":
                    pyautogui_code += f"\npyautogui.moveTo({x}, {y})"
        
        elif action_type in ["finished"]:
            pyautogui_code = f"DONE"
        
        else:
            pyautogui_code += f"\n# Unrecognized action type: {action_type}"

    return pyautogui_code


def add_box_token(input_string):
    # Step 1: Split the string into individual actions
    if "Action: " in input_string and "start_box=" in input_string:
        suffix = input_string.split("Action: ")[0] + "Action: "
        actions = input_string.split("Action: ")[1:]
        processed_actions = []
        for action in actions:
            action = action.strip()
            # Step 2: Extract coordinates (start_box or end_box) using regex
            coordinates = re.findall(r"(start_box|end_box)='\((\d+),\s*(\d+)\)'", action)
            
            updated_action = action  # Start with the original action
            for coord_type, x, y in coordinates:
                # Convert x and y to integers
                updated_action = updated_action.replace(f"{coord_type}='({x},{y})'", f"{coord_type}='<|box_start|>({x},{y})<|box_end|>'")
            processed_actions.append(updated_action)
        
        # Step 5: Reconstruct the final string
        final_string = suffix + "\n\n".join(processed_actions)
    else:
        final_string = input_string
    return final_string


from ..utils.tokenizer import get_processor, get_tokenizer

import verl.utils.torch_functional as VF
from ..models.transformers.qwen2_vl import get_rope_index

@ray.remote(num_cpus=1)
class EnvWorker():
    system_prompt = uitars_system_prompt
    
    ground_prompt = r"""Output only the coordinate of one point in your response. What element matches the following task: """

    def __init__(self, worker_idx, max_steps=15, config=None):
        self.worker_idx = worker_idx
        self.step_timeout = 60
        self.config = config
        self.use_wm = getattr(config.env, "use_wm", False)
        self.n_wm = 0
        self.n_wm_max = getattr(config.env, "n_wm_max", 5)

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
        # Always use WorldModelEnv as OSWorld is no longer used.
        print('Start to create world model env.')
        
        # Determine safe HTML save directory
        checkpoint_path = getattr(config.trainer, "save_checkpoint_path", "checkpoints")
        html_save_dir = os.path.join(checkpoint_path, "wm_htmls", f"worker_{worker_idx}")
        
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
        # self.parser = Qwen2VLParser()

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

    def reset_train_tensors(self):
        # for training
        self.input_ids = torch.zeros((0,), dtype=torch.int64)
        self.labels = torch.full((0,), -100, dtype=torch.int64)
        self.attention_mask = torch.zeros((0, ), dtype=torch.int64)

        self.pixel_values = None
        self.image_grid_thw = None

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
    
    def process_message(self, message):
        tokenizer = self.tokenizer
        processor = self.processor
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            message, return_video_kwargs=True)

        input_ids = []
        labels = []
        attention_mask = []

        image_count = 0
        pixel_values = []
        image_grid_thw = []
        for turn_idx, msg in enumerate(message):
            role = msg['role']
            content = self.load_content(msg['content'])
            prompt = f'<|im_start|>{role}\n' + content + '<|im_end|>\n'

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

        self.input_ids = torch.cat([self.input_ids, input_ids], dim=0)
        self.labels = torch.cat([self.labels, labels], dim=0)
        self.attention_mask = torch.cat([self.attention_mask, attention_mask], dim=0)

        pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values) > 0 else None
        image_grid_thw = torch.cat(image_grid_thw, dim=0) if len(image_grid_thw) > 0 else None

        if self.pixel_values is None:
            self.pixel_values = pixel_values
        else:
            if pixel_values is not None:
                self.pixel_values = torch.cat([self.pixel_values, pixel_values], dim=0)
        
        if self.image_grid_thw is None:
            self.image_grid_thw = image_grid_thw
        else:
            if image_grid_thw is not None:
                self.image_grid_thw = torch.cat([self.image_grid_thw, image_grid_thw], dim=0)
            

    def get_train_dict(self):
        position_ids = get_rope_index(
                self.processor,
                input_ids=self.input_ids,
                image_grid_thw=self.image_grid_thw,
                attention_mask=self.attention_mask,
            )
        
        input_ids, attention_mask, position_ids, labels = VF.postprocess_data(
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                position_ids=position_ids,
                max_length=self.config.data.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation='right',
                labels=self.labels
            )
        data = {
            'input_ids': input_ids,
            'labels': labels,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'n_wm': torch.tensor([self.n_wm], dtype=torch.int64),
            'traj_len': torch.tensor([self.step_counter], dtype=torch.int64)
        }
        if self.pixel_values is not None:
            multi_modal_inputs = dict()
            multi_modal_inputs['pixel_values'] = self.pixel_values
            multi_modal_inputs['image_grid_thw'] = self.image_grid_thw
            data['multi_modal_inputs'] = multi_modal_inputs
        return data
    
    def reset(self, task_config):

        self.instruction = task_config.get("instruction", None)
        self.task_config = task_config
        self.step_counter = 0
        self.is_done = False

        self.reset_train_tensors()

        trial_time = 0
        while trial_time < 8:
            try:
                obs = self.env.reset(task_config)
                break
            except Exception as e:
                print(f"Env reset exception: {e}")
                print('Env reset error: ', traceback.format_exc())
                trial_time += 1
        
        if trial_time >= 8:
            self.is_init = True
            self.is_done = True
            print('Env reset failed after 8 trials: ', task_config)
            return {
                "env_idx": self.worker_idx,
                "obs_messages": None,
                "is_done": self.is_done,
                'format_reward': 0.0
            }

        # self.agent.reset()
        self.is_init = True


        init_image = obs["screenshot"]
        # init_image = Image.open(BytesIO(init_image))

        image_base64 = base64.b64encode(BytesIO(init_image).getvalue()).decode("utf-8")

        init_messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Your are a helpful assistant."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt.format(instruction=self.instruction)
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

        self.history_images = [init_image]
        self.history_messages = init_messages

        self.process_message(init_messages)

        # since the prediction time can be very long for multienv, we pause the env to avoid automatically locking screen. 
        self.env.pause()
        return {
            'env_idx': self.worker_idx,
            'obs_messages': self.history_messages,
            'is_done': self.is_done,
            'format_reward': 0.0
        }
    
    def step(self, prediction):
        self.is_init = False
        
        # Check if the prediction contains a World Model interaction inside <think>
        # Format: <think> ... call_wm(action='...') ... </think>
        wm_call_match = re.search(r"<think>.*?call_wm\(action='(.*?)'\).*?</think>", prediction, re.DOTALL)
        
        is_imaginary = False
        if wm_call_match:
            imagined_action = wm_call_match.group(1)
            # If the prediction ends after the </think> block or contains no final Action, it's an imaginary turn
            if "Action:" not in prediction.split("</think>")[-1]:
                is_imaginary = True
                self.n_wm += 1
                print(f"Detected World Model call in <think>: {imagined_action}")

        try:
            if is_imaginary:
                # Handle imaginary turn: use World Model to get next state
                action_desc = imagined_action
                info = {"imaginary": True}
                actions = [f"Imagined: {imagined_action}"]
            else:
                # Regular turn: also use World Model for simulation
                parsed_responses = parse_action_to_structure_output(
                    prediction,
                    self.action_parse_res_factor,
                    1080, # height
                    1920, # width
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
                    
                    # Convert parsed action back to a descriptive string for WM
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
                # Execute simulation in WorldModelEnv
                obs, reward, step_done, _ = self.env.step(action_desc)
                obs_screenshot = obs['screenshot']
                
                # Convert back to bytes for consistency if it's a numpy array
                if isinstance(obs_screenshot, np.ndarray):
                    buffer = BytesIO()
                    Image.fromarray(obs_screenshot).save(buffer, format="JPEG")
                    obs_screenshot = buffer.getvalue()
            
            format_reward = 0.0
        except Exception as e:
            print('Step error: ', prediction)
            print('Error traceback: ', traceback.format_exc())
            format_reward = -1.0
            actions = ['DONE'] # error output format, stop the trajectory immediately
            obs_screenshot = self.history_images[-1]
            reward = 0.0
            step_done = True
            info = {}

        if not is_imaginary:
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
                # failed to get screenshot
                self.process_message(self.history_messages[-1:]) # gpt answer only
                return {
                    'env_idx': self.worker_idx,
                    'obs_messages': None,
                    'is_done': self.is_done,
                    'format_reward': format_reward,
                    'n_wm': self.n_wm
                }

            image_base64 = base64.b64encode(BytesIO(obs_screenshot).getvalue()).decode('utf-8')

            role = "user" if not is_imaginary else "system"
            content_prefix = "The result of your imagined action is shown below. Please continue your thinking or provide a final action." if is_imaginary else ""

            self.history_messages.append({
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": content_prefix
                    },
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_base64}",
                        "min_pixels": 3136,
                        "max_pixels": 2116800,
                    }
                ]
            })
            self.process_message(self.history_messages[-2:]) # gpt answer + next image
            return {
                'env_idx': self.worker_idx,
                'obs_messages': self.history_messages,
                'is_done': self.is_done,
                'format_reward': format_reward,
                'n_wm': self.n_wm
            }
        else:
            self.process_message(self.history_messages[-1:]) # gpt answer only
            return {
                'env_idx': self.worker_idx,
                'obs_messages': None,
                'is_done': self.is_done,
                'format_reward': format_reward,
                'n_wm': self.n_wm
            }
            
    
    def evaluate(self):
        try:
            self.env.unpause()
            # we dont care the env after evaluation, since the reset will destroy it and create new env.
            return self.env.evaluate(instruction=self.instruction)
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0
            
    
    def get_history_messages(self):
        return self.history_messages
    
    def get_history_images(self):
        return self.history_images
    
    def is_done(self):
        return self.is_done

    def is_init(self):
        return self.is_init