import json
import base64
import os
import logging
import re
from typing import Dict, List, Union
import copy
import io
from io import BytesIO
from PIL import Image

import backoff
import openai
from openai import OpenAI
from requests.exceptions import SSLError
from google.api_core.exceptions import (
    BadRequest,
    InternalServerError,
    InvalidArgument,
    ResourceExhausted,
)


from .output_parser import  Qwen2VLParser, UITarsGroundModel


logger = logging.getLogger("desktopenv.agent")

def prev_actions_to_string(prev_actions, n_prev=3):  
    result = ""  
    n_prev = min(n_prev, len(prev_actions))  # Limit n_prev to the length of the array  
    for i in range(1, n_prev + 1):  
        action = prev_actions[-i]  # Get the element at index -i (from the end)  
        result += f"Screen is currently at time step T. Below is the action executed at time step T-{i}: \n{action}\n\n"  
    return result  

from PIL import Image



class CoTQwen2VLAgent():
    system_prompt = '''\
You are STEVE, an AI that completes tasks on a user's computer with mouse and keyboard operations. Your role as an assistant involves thoroughly exploring the user task through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of screenshot analysis, action reflection, task planning and error backtracing to complete the user task step-by-step. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> thought <|end_of_thought|> Each step should include detailed considerations such as screenshot analysis, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {thoughts, rationale, decision, python(optional)} <|end_of_solution|>"""

You can use the following functions:
```python
computer.mouse.move(<|object_ref_start|>"target_element"<|object_ref_end|><|point_start|>(x,y)<|point_end|>)
computer.mouse.single_click()
computer.mouse.double_click()
computer.mouse.right_click() 
computer.mouse.scroll(dir="up/down")
computer.mouse.drag(<|object_ref_start|>"target_element"<|object_ref_end|><|point_start|>(x,y)<|point_end|>) # drag from current cursor position to the element

computer.keyboard.write("text")
computer.keyboard.press("key") # press a key once (donw and up)
computer.keyboard.keyDown("key") # press and hold a key
computer.keyboard.keyUp("key") # release a key
computer.keyboard.hotkey("key1", "key2", ...) # press multiple keys simultaneously to trigger a shortcut

```
    '''
    def __init__(
            self,
            server: str = "http://10.1.1.3:9000/v1",
            model: str = "cot_qwen2vl",
            grounding_server: str = "http://10.1.1.3:8001/v1",
            grounding_model: str = "ui-tars",
            obs_view = "screen", # "screen" or "window"
            auto_window_maximize = False,
            temperature: float = 0.5,
    ):
        self.action_space = "pyautogui"
        self.server = server
        self.model = model
        
        print(self.model, self.server)
        self.obs_view = obs_view
        self.auto_window_maximize = auto_window_maximize
        self.prev_window_title = None
        self.prev_window_rect = None

        self.client = OpenAI(
            base_url=server,
            api_key="empty",
        )

        self.parser = Qwen2VLParser()
        print(grounding_server, grounding_model)
        self.ground_model = UITarsGroundModel(
            server=grounding_server,
            model=grounding_model,
            input_format='qwen2vl'
        )

        self.n_prev = 15
        self.step_counter = 0

        self.history_images = []
        self.history_images = []
        self.history_messages = []
    
    def reset_server(self, server: str):
        self.client = OpenAI(
            base_url=server,
            api_key="empty",
        )
        self.server = server

    def encode_image(self, image: Union[str, Image.Image]) -> str:  
        if isinstance(image, str):  
            with open(image, "rb") as image_file:  
                return base64.b64encode(image_file.read()).decode("utf-8")  
        elif isinstance(image, Image.Image):  
            buffer = io.BytesIO()  
            # convert to rgb if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(buffer, format="JPEG")  
            return base64.b64encode(buffer.getvalue()).decode("utf-8")  
  
    def get_url_payload(self, url: str) -> dict:  
        return {  
            "type": "image_url",  
            "image_url": {  
                "url": url  
            }  
        }  
  
    def get_base64_payload(self, base64_image: str, detail="auto") -> dict:  
        return {  
            "type": "image_url",  
            "image_url": {  
                "url": f"data:image/jpeg;base64,{base64_image}",  
                "detail": detail 
            }  
        }  
    

    def predict(self, instruction: str, obs: Dict, last_action_after_obs=None) -> List:
        """
        Predict the next action(s) based on the current observation.
        """
        logs={}
        
        # image_file = BytesIO(obs['screenshot'])
        base64_image = obs['screenshot']
        view_image = Image.open(BytesIO(base64_image))

        logger.info(f"Thinking...")
        
        # check if the last message is from the assistant
        if len(self.history_messages) > 0 and self.history_messages[-1]['role'] != 'assistant':
            logger.info(f"Error in history_message. Seems like the last message is not from the assistant. Remove the last message.")
            self.history_messages = self.history_messages[:-1]
                    
        message_context = copy.deepcopy(self.history_messages[-self.n_prev*2:])

        query_message = {
            "role": "user",
            "content": [
                self.get_base64_payload(self.encode_image(view_image)),
                {"type": "text", "text": f'Task: {instruction}. Please generate the next move.'}
            ],
        }

        message_context.append(copy.deepcopy(query_message))
        self.history_messages.append(copy.deepcopy(query_message))

        # add system_prompt
        message_context[0]["content"].insert(0, {"type": "text", "text": self.system_prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=message_context,
            frequency_penalty=0.2, # do not use too large value because it will result in action format error
            temperature=0.6,
            max_tokens=8192,
            extra_body={"skip_special_tokens": False}
        )
        plan_result_full = response.choices[0].message.content
        # print(plan_result_full)

        plan_result = re.search(r'<\|begin_of_solution\|>(.*)<\|end_of_solution\|>', plan_result_full, re.DOTALL)
        if plan_result:
            plan_result = plan_result.group(1)
        else:
            logger.info(f"Error in plan_result. Cannot find the plan_result.")
            plan_result = plan_result_full
        

        decision_block, actions = self.parser.parse(plan_result)
        if actions in ['DONE', 'FAIL', 'WAIT', 'CALL_USER']:
            actions_grounded = actions
        else:
            actions_grounded, positions = self.ground_model.ground_action(view_image, actions)

        logs['plan_result_full'] = plan_result_full
        logs['actions'] = actions
        logs['actions_grounded'] = actions_grounded
        logs['plan_result'] = plan_result

        self.step_counter += 1
        self.history_images.append(view_image)
        self.history_actions.append(plan_result)
        self.history_messages.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": plan_result_full}
            ]
        })

        # remap action space to OSWorld pyautogui
        if actions_grounded == 'CALL_USER':
            actions_grounded = "FAIL"
        
        actions_osworld = self.remap_action_space(actions_grounded, view_image.width, view_image.height)

        actions = [actions_osworld] # to list[str]
        if len(self.history_images) > self.n_prev:
            actions = ["FAIL"]
        return plan_result_full, actions, logs
        
    # @backoff.on_exception(
    #     backoff.constant,
    #     # here you should add more model exceptions as you want,
    #     # but you are forbidden to add "Exception", that is, a common type of exception
    #     # because we want to catch this kind of Exception in the outside to ensure each example won't exceed the time limit
    #     (
    #         # General exceptions
    #         SSLError,
    #         # OpenAI exceptions
    #         openai.RateLimitError,
    #         openai.BadRequestError,
    #         openai.InternalServerError,
    #         # Google exceptions
    #         InvalidArgument,
    #         ResourceExhausted,
    #         InternalServerError,
    #         BadRequest,
    #         # Groq exceptions
    #         # todo: check
    #     ),
    #     interval=30,
    #     max_tries=10,
    # )

    def remap_action_space(self, actions, width, height):
        pyautogui_code = f"import pyautogui\nimport time\n"

        def remap_mouse_functions(actions):
            drag_pattern = r"computer\.mouse\.drag\(x=(-?\d+\.\d+),\s*y=(-?\d+\.\d+)\)"
            def replace_drag(match):
                x, y = float(match.group(1)), float(match.group(2))
                x_scaled, y_scaled = int(x * width), int(y * height)
                return f"pyautogui.dragTo({x_scaled}, {y_scaled}, button='left', duration=1)"
            
            move_pattern = r"computer\.mouse\.move_abs\(x=(-?\d+\.\d+),\s*y=(-?\d+\.\d+)\)"
            def replace_move(match):
                x, y = float(match.group(1)), float(match.group(2))
                x_scaled, y_scaled = int(x * width), int(y * height)
                return f"pyautogui.moveTo({x_scaled}, {y_scaled})"
            
            actions = re.sub(drag_pattern, replace_drag, actions)
            actions = re.sub(move_pattern, replace_move, actions)

            click_pattern = r"computer\.mouse\.single_click\(\)"
            double_click_pattern = r"computer\.mouse\.double_click\(\)"
            right_click_pattern = r"computer\.mouse\.right_click\(\)"

            actions = re.sub(click_pattern, "pyautogui.click()", actions)
            actions = re.sub(double_click_pattern, "pyautogui.doubleClick()", actions)
            actions = re.sub(right_click_pattern, "pyautogui.rightClick()", actions)

            scroll_pattern = r"computer\.mouse\.scroll\(dir=\"(up|down)\"\)"
            def replace_scroll(match):
                if match.group(1) == "up":
                    return "pyautogui.scroll(200)"
                else:
                    return "pyautogui.scroll(-200)"
            actions = re.sub(scroll_pattern, replace_scroll, actions)

            return actions
    
        def remap_keyboard_functions(actions):
            keyboard_write_pattern = r'computer\.keyboard\.write\(r?(["\'])(.*?)\1\)'
            keyboard_press_pattern = r'computer\.keyboard\.press\((["\'])(.*?)\1\)'
            keyboard_keydown_pattern = r'computer\.keyboard\.keyDown\((["\'])(.*?)\1\)'
            keyboard_keyup_pattern = r'computer\.keyboard\.keyUp\((["\'])(.*?)\1\)'
            # keyboard_hotkey_pattern = r'computer\.keyboard\.hotkey\((["\'])(.*?)\1(?:,\s*(["\'])(.*?)\3)*\)'

            def replace_write(match):
                text = match.group(1)
                return f"pyautogui.write(\"{text}\", interval=0.1)"
            
            def replace_hotkey(match):
                keys = [match.group(2)]  # First key
                keys.extend(re.findall(r'(["\'])(.*?)\1', match.string[match.start():match.end()])[1:])  # Other keys
                key_list = ", ".join(f'"{key[1]}"' for key in keys)
                return f"pyautogui.hotkey({key_list})"

            code_snippet = actions
            code_snippet = re.sub(keyboard_write_pattern, r'pyautogui.write(\1\2\1)', code_snippet)
            code_snippet = re.sub(keyboard_press_pattern, r'pyautogui.press(\1\2\1)', code_snippet)
            code_snippet = re.sub(keyboard_keydown_pattern, r'pyautogui.keyDown(\1\2\1)', code_snippet)
            code_snippet = re.sub(keyboard_keyup_pattern, r'pyautogui.keyUp(\1\2\1)', code_snippet)
            # code_snippet = re.sub(keyboard_hotkey_pattern, replace_hotkey, code_snippet)
            code_snippet = code_snippet.replace("computer.keyboard.hotkey", "pyautogui.hotkey")

            actions = code_snippet
            return actions

        actions = remap_mouse_functions(actions)
        actions = remap_keyboard_functions(actions)
        return actions


    def reset(self, runtime_logger):
        self.prev_window_title = None
        self.prev_window_rect = None
        self.clipboard_content = None
        self.step_counter = 0
        self.history_images = []
        self.history_actions = []
        self.history_messages = []
    
    