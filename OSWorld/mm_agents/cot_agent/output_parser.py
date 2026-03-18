import re
import base64
import random
import traceback

from openai import OpenAI

from PIL import Image, ImageDraw
import io
from typing import Union


def remove_min_leading_spaces(text):  
    lines = text.split('\n')  
    min_spaces = min(len(line) - len(line.lstrip(' ')) for line in lines if line)  
    return '\n'.join([line[min_spaces:] for line in lines])  

def encode_image(image: Union[str, Image.Image]) -> str:  
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


class UITarsGroundModel():
    def __init__(self, server='10.1.1.3:8000', model='ui-tars', input_format='gpt4o', ground_type='point'):
        self.server = server
        self.model = model

        assert model in ['ui-tars', 'mgm', 'text']
        assert input_format in ['gpt4o', 'qwen2vl']
        # if using ui-tars with qwen2vl, the text x,y output of qwen2vl will be ignored

        self.input_format = input_format
        self.ground_type = ground_type

        self.client = OpenAI(
            # base_url="http://10.1.1.3:8003/v1",
            base_url=server,
            api_key="empty",
        )

    
    def ground_action(self, image, action):
        matches = re.findall(r'computer.mouse.(move|drag)\(["\'](.*?)["\']\)', action)
        if not matches:
            return action, []

        image_base64 = encode_image(image)
        positions = []


        try:
            for action_type, ref_ui in matches:
                if self.input_format == 'gpt4o':
                    ref_ui_text = ref_ui
                    xc, yc = self.ground_single_action(image_base64, ref_ui)

                elif self.input_format == 'qwen2vl':
                    ref_ui_text = re.search(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>', ref_ui).groups()[0]
                    xc, yc = re.search(r'<\|box_start\|>\((\d+),(\d+)\)<\|box_end\|>', ref_ui).groups()
                    xc, yc = int(xc) / 1932, int(yc) / 1092
                    # xc, yc = re.search(r'<\|point_start\|>\((\d+),(\d+)\)<\|point_end\|>', ref_ui).groups()
                    # xc, yc = int(xc) / 1000, int(yc) / 1000
                    if self.model in ['ui-tars', ]:
                        xc, yc = self.ground_single_action(image_base64, ref_ui_text) # re-ground with two stage grounding model (UITars)

                if action_type == 'move':
                    replace_action = f'computer.mouse.move_abs(x={xc}, y={yc})'
                elif action_type == 'drag':
                    replace_action = f'computer.mouse.drag(x={xc}, y={yc})'

                positions.append((xc, yc))
                action = re.sub(r'computer.mouse.(move|drag)\(["\']' + re.escape(ref_ui) + r'["\']\)', 
                    replace_action, action)

        except Exception as e:
            print('Error in grounding: ', e)
            traceback.print_exc()
            return action, []

        return action, positions
    
    def ground_single_action(self, image_base64, ref_ui):
        if self.model == 'ui-tars':
            return self.ground_single_action_utars(image_base64, ref_ui)
        elif self.model == 'mgm':
            return self.ground_single_action_mgm(image_base64, ref_ui)
    
    def ground_single_action_utars(self, image_base64, ref_ui):
        # prompt_simple = r"""Output only the coordinate of one point in your response. What element matches the following task: """
        prompt_simple = r"""Output only the coordinate of one point in your response. Do not respond "there are none". What element matches the following task: """
        for i in range(10): # max try 10 times
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                            {"type": "text", "text": prompt_simple + ref_ui}
                        ],
                    },
                ],
                frequency_penalty=1,
                max_tokens=256,
                temperature=random.choice([0.1, 0.3]),
            )
            text = response.choices[0].message.content
            print('grounder: ', ref_ui, text)
            # (x,y) 
            match = re.search(r'\((\d+),(\d+)\)', text)
            if match:
                x, y = map(int, match.groups())
                x, y = x / 1920, y / 1080
                # x, y = x / 1000, y / 1000
                return (x, y)
        
        raise Exception('Cannot ground the action')

    
    def ground_single_action_mgm(self, image_base64, ref_ui):
        # the box is in ref_ui in text format
        # bbox_match = re.search(r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>', ref_ui)

        # use the gronding model and the reference expression to get bbox
        # prompt_simple = r"""Output only the coordinate of one point in your response. What element matches the following task: """
        prompt_simple = f'Provide the center point for the UI element: "{ref_ui}" in the image.'

        for i in range(10): # max try 10 times
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                            {"type": "text", "text": prompt_simple}
                        ],
                    },
                ],
                frequency_penalty=1,
                max_tokens=256,
                temperature=0.1,
            )
            text = response.choices[0].message.content
            # (x,y) 
            match = re.search(r'>\((\d+),(\d+)\)<', text)
            if match:
                x, y = map(int, match.groups())
                x, y = x / 1000, y / 1000
                return (x, y)

        raise Exception('Cannot ground the action')


class Qwen2VLParser():

    def parse(self, plan_result):
        code_block = re.search(r'```python\n(.*?)```', plan_result, re.DOTALL)
        if code_block:
            code_block_text = code_block.group(1)
            code_block_text = remove_min_leading_spaces(code_block_text)
            actions = code_block_text
        else:
            actions = 'Not found'
        
        # extract the high-level decision block
        decision_block = re.search(r'```decision\n(.*?)```', plan_result, re.DOTALL)
        if decision_block:
            decision_block_text = decision_block.group(1)
            if "DONE" in decision_block_text:
                actions = "DONE"
            elif "FAIL" in decision_block_text:
                actions = "FAIL"
            elif "WAIT" in decision_block_text:
                actions = "WAIT"
            elif "CALL_USER" in decision_block_text:
                actions = "CALL_USER"
        else:
            decision_block_text = 'Not found'
            actions = 'FAIL'

        return decision_block_text, actions



class GPT4oParser():


    def parse(self, plan_result):

        code_block = re.search(r'```python\n(.*?)```', plan_result, re.DOTALL)
        if code_block:
            code_block_text = code_block.group(1)
            code_block_text = remove_min_leading_spaces(code_block_text)
            actions = code_block_text
        
        # extract the high-level decision block
        decision_block = re.search(r'```decision\n(.*?)```', plan_result, re.DOTALL)
        if decision_block:
            decision_block_text = decision_block.group(1)
            if "DONE" in decision_block_text:
                actions = "DONE"
            elif "FAIL" in decision_block_text:
                actions = "FAIL"
            elif "WAIT" in decision_block_text:
                actions = "WAIT"
            elif "CALL_USER" in decision_block_text:
                actions = "CALL_USER"
        else:
            pass

        return decision_block_text, actions

if __name__ == '__main__':

    ground_model = UTarsGroundModel()
    image = "./image_for_test.jpg"
    image = Image.open(image)

    action = "```python\ncomputer.mouse.move(\"three dot settings menu of the browser\")```"
    action = "```python\ncomputer.mouse.drag(\"maxmize the window\")```"

    grounded_action, positions = ground_model.ground_action(image, action)
    draw = ImageDraw.Draw(image)
    for x, y in positions:
        x, y = x * image.width, y * image.height
        draw.rectangle((x-5, y-5, x+5, y+5), outline='red', width=3)

    print(grounded_action)
    image.save('./visual_test.png')