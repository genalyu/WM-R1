"""
World Model HTTP server — uses transformers natively (supports Qwen3-VL).
Provides OpenAI-compatible API at /v1/chat/completions.

Start one per compute node. EnvWorkers on the same node call this via HTTP.
"""
import os
import re
import json
import time
import base64
import io
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from datetime import datetime
from threading import Lock

import numpy as np
import torch
from PIL import Image
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
    _lock = None  # Prevent concurrent inference to avoid GPU OOM

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        elif self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "object": "list",
                "data": [{"id": "qwen3-vl-wm", "object": "model"}]
            }).encode())

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(content_length))

            try:
                with self._lock:
                    output_text = self.infer_from_openai_messages(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    self.wfile.write(json.dumps({
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": "qwen3-vl-wm",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": output_text},
                            "finish_reason": "stop",
                        }],
                    }).encode())
                except (BrokenPipeError, ConnectionResetError):
                    print(f"[WARN] client disconnected before response (idx={body.get('idx', '?')})")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

    def _decode_image_from_messages(self, content_list):
        """Extract image from OpenAI-format message content."""
        for item in content_list:
            if item.get("type") == "image_url":
                url = item["image_url"]["url"]
                if url.startswith("data:"):
                    b64_data = url.split(",", 1)[1]
                    return Image.open(io.BytesIO(base64.b64decode(b64_data)))
            elif item.get("type") == "image" and "image" in item:
                img = item["image"]
                if isinstance(img, Image.Image):
                    return img
                elif isinstance(img, (bytes, bytearray)):
                    return Image.open(io.BytesIO(img))
        return None

    def infer_from_openai_messages(self, body):
        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 2048)
        temperature = body.get("temperature", 0.0)

        # Reconstruct messages for Qwen3VL processor
        qwen_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])

            # Convert OpenAI image_url format to Qwen3VL format
            if isinstance(content, list):
                converted_content = []
                for item in content:
                    if item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            b64_data = url.split(",", 1)[1]
                            image = Image.open(io.BytesIO(base64.b64decode(b64_data)))
                            converted_content.append({"type": "image", "image": image})
                    elif item.get("type") == "text":
                        converted_content.append({"type": "text", "text": item["text"]})
                    else:
                        converted_content.append(item)
                qwen_messages.append({"role": role, "content": converted_content})
            else:
                qwen_messages.append({"role": role, "content": content})

        # Apply chat template and generate (Qwen3VL handles image processing internally)
        inputs = self.processor.apply_chat_template(
            qwen_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            use_cache=True,           # 启用 KV cache
            do_sample=temperature > 0,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        # Explicitly clear KV cache to avoid memory accumulation across requests
        if hasattr(self.model, "past_key_values") and self.model.past_key_values is not None:
            self.model.past_key_values = None
        torch.cuda.empty_cache()
        return output_text

    # Legacy endpoint for backward compatibility
    def do_POST_legacy(self):
        content_length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(content_length))

        image_bytes = base64.b64decode(body["image_b64"])
        image = Image.open(io.BytesIO(image_bytes))
        goal = body.get("goal", body.get("description", ""))
        description = body.get("description", "")
        action = body.get("action", "")
        html_dir = body.get("html_dir", self.html_dir)
        idx = body.get("idx", 0)

        try:
            html_path, png_path = self.infer(goal, description, action, image, html_dir, idx)
            with open(png_path, "rb") as f:
                png_b64 = base64.b64encode(f.read()).decode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"png_b64": png_b64}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def infer(self, goal, description, action, image, html_dir, idx):
        user_text = USER_PROMPT.format(goal=goal, description=description, action=str(action))
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
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
        html_path = os.path.join(html_dir, f"test_{idx}.html")
        png_path = os.path.join(html_dir, f"test_{idx}.png")

        with open(html_path, "w") as f:
            f.write(clean_html)

        render_aligned_png(html_path, png_path)
        return html_path, png_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--port", type=int, default=18888)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Loading Qwen3-VL from {args.model}...")
    WMHandler.processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Limit GPU memory to avoid OOM from large KV cache during generation
    total_mem = torch.cuda.get_device_properties(0).total_memory
    max_memory = {0: int(total_mem * args.gpu_memory_utilization)}
    print(f"GPU memory limit: {max_memory[0] / 1e9:.1f} GB ({args.gpu_memory_utilization:.0%} of {total_mem / 1e9:.1f} GB)")

    WMHandler.model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        device_map=args.device,
        trust_remote_code=True,
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,        # 使用 bf16 而不是 fp32
        low_cpu_mem_usage=True,            # 降低 CPU 内存使用
        use_cache=True,                    # 启用 KV cache
    )
    WMHandler._lock = Lock()
    print("Model loaded.")

    server = type("ThreadedHTTPServer", (ThreadingMixIn, HTTPServer), {"daemon_threads": True})(("0.0.0.0", args.port), WMHandler)
    print(f"WM server running on port {args.port}")
    server.serve_forever()
