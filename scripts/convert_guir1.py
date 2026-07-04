#!/usr/bin/env python3
"""Convert fansunqi/guir1_train_split to WM-R1 JSONL (mobile only).

Usage:
    python scripts/convert_guir1.py --output data/guir1/

Output:
    data/guir1/train.jsonl
    data/guir1/screenshots/*.png
"""
import argparse
import json
import os
import sys
from io import BytesIO
from PIL import Image


TARGET_WIDTH = 1080
TARGET_HEIGHT = 2400


def resize_if_needed(img: Image.Image) -> Image.Image:
    if img.size == (TARGET_WIDTH, TARGET_HEIGHT):
        return img
    return img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)


def main():
    parser = argparse.ArgumentParser(description="Convert GUI-R1 mobile data to WM-R1 JSONL")
    parser.add_argument("--dataset", type=str, default="fansunqi/guir1_train_split",
                        help="HuggingFace dataset name")
    parser.add_argument("--output", type=str, default="data/guir1",
                        help="Output directory")
    parser.add_argument("--platform", type=str, default="mobile",
                        help="Filter by platform: 'mobile', 'desktop', 'web', or 'all'")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print(f"Loading {args.dataset} (split={args.split})...")
    ds = load_dataset(args.dataset, split=args.split)
    print(f"Loaded {len(ds)} examples")

    # Inspect columns
    print(f"Columns: {ds.column_names}")
    print(f"First example keys: {list(ds[0].keys())}")

    # Show first example structure
    sample = ds[0]
    for k, v in sample.items():
        vtype = type(v).__name__
        vstr = str(v)[:150] if not isinstance(v, (bytes, Image.Image, dict)) else f"[{vtype}]"
        print(f"  {k}: {vtype} = {vstr}")

    os.makedirs(args.output, exist_ok=True)
    screenshot_dir = os.path.join(args.output, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    records = []
    skipped = 0
    platform_counts = {}

    for idx, example in enumerate(ds):
        try:
            # --- Filter by platform ---
            platform = str(example.get("task_type", example.get("platform", ""))).lower()
            platform_counts[platform] = platform_counts.get(platform, 0) + 1

            if args.platform != "all" and args.platform not in platform:
                skipped += 1
                continue

            # --- Extract instruction ---
            instruction = example.get("instruction", example.get("goal", example.get("task", "")))
            if not instruction:
                skipped += 1
                continue

            # --- Extract image ---
            img = None
            img_field = example.get("image", None)

            if isinstance(img_field, Image.Image):
                img = img_field
            elif isinstance(img_field, dict):
                if "bytes" in img_field and img_field["bytes"]:
                    img = Image.open(BytesIO(img_field["bytes"]))
                elif "path" in img_field and img_field["path"]:
                    img = Image.open(img_field["path"])
            elif isinstance(img_field, bytes):
                img = Image.open(BytesIO(img_field))
            elif isinstance(img_field, str) and os.path.exists(img_field):
                img = Image.open(img_field)

            if img is None:
                skipped += 1
                continue

            img = img.convert("RGB")
            img = resize_if_needed(img)

            # --- Save screenshot ---
            task_id = example.get("id", f"guir1_{idx}")
            img_filename = f"guir1_{task_id}_0.png"
            img.save(os.path.join(screenshot_dir, img_filename), "PNG")

            # --- Create JSONL record ---
            record = {
                "task_id": f"guir1_{task_id}",
                "instruction": instruction,
                "domain": "android",
                "traj": [
                    {"image": f"screenshots/{img_filename}"}
                ]
            }
            records.append(record)

            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(ds)}, mobile records: {len(records)}")

        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  Skipped {idx}: {e}")

    # --- Write JSONL ---
    jsonl_path = os.path.join(args.output, "train.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n=== Results ===")
    print(f"Platform distribution: {platform_counts}")
    print(f"Mobile records: {len(records)} → {jsonl_path}")
    print(f"Skipped (non-mobile/missing): {skipped}")
    print(f"Screenshots: {screenshot_dir}")


if __name__ == "__main__":
    main()
