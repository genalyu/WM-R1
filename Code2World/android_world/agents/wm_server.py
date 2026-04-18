"""
World Model HTTP server — runs in a separate conda env with high transformers version.
Start one per compute node. EnvWorkers on the same node call this via HTTP.
"""
import os
import base64
import io
import json
import argparse
import re
from PIL import Image
from http.server import HTTPServer, BaseHTTPRequestHandler

# This env needs transformers 4.57.0+
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from android_world.agents.wm_utils import render_aligned_png


SYSTEM_PROMPT = """You are an expert **UI State Transition Simulator** and **Frontend Developer**.
Your task is to predict the **NEXT UI STATE** based on a screenshot of the current state and a user interaction.

### Rules
- Analyze the action. If the user clicks a button, show the result.
- Output ONLY raw HTML. Start with `<!DOCTYPE html>` and end with `</html>`.
- All visible content MUST be wrapped in `<div id="render-target"> ... </div>`
  with `width: 1080px; height: 2400px; position: relative; overflow: hidden;`.
- The `<body>` tag must have `margin: 0; padding: 0; background: transparent;`.
"""

USER_PROMPT = "<image>\n### INPUT CONTEXT\n1. **User Intent**: {goal}\n2. **Interaction Description**: {description}\n3. **Action Data**: {action}\n\n### COMMAND\nBased on the visual cues and interaction data, generate the HTML for the RESULTING UI STATE.\n"


class WMHandler(BaseHTTPRequestHandler):
    model = None
    processor = None
    html_dir = "/tmp/wm_htmls"

    def log_message(self, *args):  # noqa: ARG002
        pass  # Suppress request logs

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        image_bytes = base64.b64decode(data['image_b64'])
        image = Image.open(io.BytesIO(image_bytes))

        goal = data.get('goal', data.get('description', ''))
        description = data.get('description', '')
        action = data.get('action', '')
        html_dir = data.get('html_dir', self.html_dir)
        idx = data.get('idx', 0)

        try:
            html_path, png_path = self.infer(goal, description, action, image, html_dir, idx)
            with open(png_path, 'rb') as f:
                png_b64 = base64.b64encode(f.read()).decode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'png_b64': png_b64}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())

    def infer(self, goal, description, action, image, html_dir, idx):
        user_text = USER_PROMPT.format(goal=goal, description=description, action=str(action))
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Extract clean HTML
        text = output_text.replace("```html", "").replace("```", "")
        start_match = re.search(r"<!DOCTYPE html>", text, re.IGNORECASE)
        end_match = re.search(r"</html>", text, re.IGNORECASE)
        clean_html = text[start_match.start():end_match.end()] if start_match and end_match else text.strip()

        os.makedirs(html_dir, exist_ok=True)
        html_path = os.path.join(html_dir, f'test_{idx}.html')
        png_path = os.path.join(html_dir, f'test_{idx}.png')

        with open(html_path, 'w') as f:
            f.write(clean_html)

        render_aligned_png(html_path, png_path)
        return html_path, png_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--port', type=int, default=18888)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    print(f"Loading Qwen3-VL from {args.model}...")
    WMHandler.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    WMHandler.model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, device_map=args.device, trust_remote_code=True
    )
    print("Model loaded.")

    server = HTTPServer(('0.0.0.0', args.port), WMHandler)
    print(f"WM server running on port {args.port}")
    server.serve_forever()
