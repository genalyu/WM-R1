"""
Debug script for Qwen2.5-VL image_grid_thw vs pixel_values mismatch.
Supports AgentNet JSONL format where images are in traj[0].image.

Usage:
    python debug_image_mismatch.py \
        --data-path /path/to/data.jsonl \
        --model-path /path/to/Qwen2.5-VL-7B \
        --image-dir /path/to/screenshots/
"""

import argparse
import json
import math
import os

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default='/public/home/xlwang/genalyu/dataset/agentnet/agentnet_test_1k.jsonl', required=True, help="Path to JSONL data file")
    parser.add_argument("--model-path", type=str, default='/public/home/xlwang/genalyu/models/Qwen2.5-VL-3B-Instruct', required=True, help="Path to Qwen2.5-VL model")
    parser.add_argument("--image-dir", type=str, required=True,default='/public/home/xlwang/genalyu/dataset/agentnet',
                        help="Directory where screenshot images are stored")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to check")
    parser.add_argument("--max-pixels", type=int, default=768 * 768)
    parser.add_argument("--min-pixels", type=int, default=4 * 4)
    return parser.parse_args()


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def process_image(image, max_pixels, min_pixels):
    """Same logic as ImageProcessMixin in osworld.py"""
    orig_w, orig_h = image.size
    orig_pixels = orig_w * orig_h
    print(f"    Original: {orig_w}x{orig_h} = {orig_pixels}px")

    if (orig_w * orig_h) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (orig_w * orig_h))
        w, h = int(orig_w * resize_factor), int(orig_h * resize_factor)
        image = image.resize((w, h))
        print(f"    After max_pixels resize: {w}x{h} = {w*h}px")

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        w, h = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((w, h))
        print(f"    After min_pixels resize: {w}x{h} = {w*h}px")

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def compute_expected_tokens_from_grid(image_grid_thw, merge_size=2):
    """Compute expected visual tokens from image_grid_thw."""
    total = 0
    for t, h, w in image_grid_thw.tolist():
        total += t * (h // merge_size) * (w // merge_size)
    return total


def main():
    args = parse_args()

    # Load data
    print(f"Loading JSONL from {args.data_path}...")
    dataset = load_jsonl(args.data_path)
    print(f"Total samples: {len(dataset)}")

    # Show first row keys
    sample_row = dataset[0]
    print(f"Available keys: {list(sample_row.keys())}")

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    print(f"spatial_merge_size = {merge_size}")

    num_crashes = 0
    num_mismatches = 0
    num_ok = 0
    num_skipped = 0

    for idx in range(min(args.num_samples, len(dataset))):
        row = dataset[idx]
        print(f"\n{'='*60}")
        print(f"Sample {idx} (task_id={row.get('task_id', '?')})")
        print(f"{'='*60}")

        # Load image from traj[0].image
        traj = row.get("traj", [])
        if not traj or "image" not in traj[0]:
            print("  No traj[0].image found, skipping.")
            num_skipped += 1
            continue

        image_filename = traj[0]["image"]
        image_path = os.path.join(args.image_dir, image_filename)
        if not os.path.isfile(image_path):
            # Try images/ subdir
            image_path = os.path.join(args.image_dir, "images", image_filename)
        if not os.path.isfile(image_path):
            print(f"  Image not found: {image_path}, skipping.")
            num_skipped += 1
            continue

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"  Failed to load image: {e}, skipping.")
            num_skipped += 1
            continue

        # Process image (custom resize)
        processed = process_image(img, args.max_pixels, args.min_pixels)
        pw, ph = processed.size
        print(f"    Final processed: {pw}x{ph}")

        # Build prompt
        instruction = row.get("instruction", row.get("natural_language_task", ""))
        content_list = [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]
        messages = [{"role": "user", "content": content_list}]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # ---- Method A: processor with do_resize=False ----
        # Skipped: transformers 4.51.0 has a bug with do_resize=False
        print(f"\n  --- Method A: SKIPPED (do_resize=False not supported in 4.51.0) ---")

        # ---- Method B: processor with default behavior ----
        print(f"\n  --- Method B: processor (default behavior) ---")
        try:
            model_inputs_b = processor(
                [processed], [prompt],
                add_special_tokens=False,
                return_tensors="pt",
            )
        except TypeError:
            print("    do_resize kwarg NOT supported, using default")
            model_inputs_b = processor(
                [processed], [prompt],
                add_special_tokens=False,
                return_tensors="pt",
            )

        grid_b = model_inputs_b["image_grid_thw"]
        pixels_b = model_inputs_b["pixel_values"]
        expected_b = compute_expected_tokens_from_grid(grid_b, merge_size)
        print(f"    image_grid_thw: {grid_b.tolist()}")
        print(f"    pixel_values shape: {pixels_b.shape}")
        print(f"    Expected tokens from grid_thw: {expected_b}")
        if pixels_b.dim() == 4:
            actual_b = pixels_b.shape[2]
            print(f"    Actual tokens (from pixel_values num_patches): {actual_b}")
        else:
            actual_b = pixels_b.shape[0]
            print(f"    Actual tokens (flat): {actual_b}")
        mismatch_b = expected_b != actual_b
        print(f"    Mismatch: {mismatch_b}")

        # ---- Forward pass ----
        print(f"\n  --- Forward pass test ---")
        use_grid = grid_b
        use_pixels = pixels_b

        try:
            visual = model.visual
            device = visual.get_device()

            # pixel_values shape: [1, channels, num_patches, patch_dim]
            # Flatten to [num_patches, patch_dim] for visual encoder
            pixels_flat = use_pixels.squeeze(0).to(device)
            if pixels_flat.dim() == 3:
                pixels_flat = pixels_flat.reshape(-1, pixels_flat.shape[-1])

            print(f"    Input to visual: pixels={pixels_flat.shape}, grid_thw={use_grid.to(device)}")

            with torch.no_grad():
                hidden = visual(pixels_flat, grid_thw=use_grid.to(device))

            actual_size = hidden.shape[0] * hidden.shape[-1] if hidden.dim() == 2 else hidden.numel()
            expected_tokens = compute_expected_tokens_from_grid(use_grid, merge_size)
            expected_size = expected_tokens * hidden.shape[-1]

            print(f"    Forward output shape: {hidden.shape}")
            print(f"    Expected size (grid_tokens * hidden_dim): {expected_size}")
            print(f"    Actual size: {actual_size}")

            if actual_size != expected_size:
                print(f"    *** MISMATCH DETECTED ***")
                print(f"    grid_thw predicts {expected_tokens} tokens")
                print(f"    visual encoder produced {hidden.shape[0]} tokens")
                num_mismatches += 1
            else:
                print(f"    OK")
                num_ok += 1

        except RuntimeError as e:
            print(f"    *** CRASH: {e} ***")
            num_crashes += 1

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  OK:         {num_ok}")
    print(f"  Mismatch:   {num_mismatches}")
    print(f"  Crash:      {num_crashes}")
    print(f"  Skipped:    {num_skipped}")

    if num_mismatches > 0 or num_crashes > 0:
        print(f"\nDiagnosis:")
        if num_mismatches > 0:
            print(f"  -> image_grid_thw 和 pixel_values 不匹配")
            print(f"  -> 可能是 processor 内部 resize 和自定义 resize 冲突")
        if num_crashes > 0:
            print(f"  -> transformers 库 visual encoder forward 直接 crash")
            print(f"  -> 可能是 transformers 版本 bug")
        print(f"\nRecommendation:")
        print(f"  1. 移除 ImageProcessMixin，完全交给 processor 内部 resize")
        print(f"  2. 或者确保 ImageProcessMixin 输出的尺寸是 14 的倍数")
    else:
        print(f"\nAll samples OK. Crash may be triggered by a specific edge-case image.")


if __name__ == "__main__":
    main()
