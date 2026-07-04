#!/usr/bin/env python3
"""Convert local train_mobile.parquet to WM-R1 JSONL.

Usage:
    python scripts/convert_guir1.py --input data/train_mobile.parquet --output data/guir1/
"""
import argparse
import json
import os
import sys
from io import BytesIO
from PIL import Image
import pandas as pd


TARGET_WIDTH = 1080
TARGET_HEIGHT = 2400


def resize_if_needed(img: Image.Image) -> Image.Image:
    if img.size == (TARGET_WIDTH, TARGET_HEIGHT):
        return img
    return img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)


def extract_image(value):
    """Try to extract PIL.Image from various formats."""
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, bytes):
        return Image.open(BytesIO(value))
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"]:
            return Image.open(BytesIO(value["bytes"]))
        if "path" in value and value["path"] and os.path.exists(value["path"]):
            return Image.open(value["path"])
    if isinstance(value, str) and os.path.exists(value):
        return Image.open(value)
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert train_mobile.parquet to WM-R1 JSONL")
    parser.add_argument("--input", type=str, default="data/train_mobile.parquet",
                        help="Input parquet file")
    parser.add_argument("--output", type=str, default="data/guir1",
                        help="Output directory")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found")
        sys.exit(1)

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Show first row structure
    print("\nFirst row:")
    for col in df.columns:
        val = df.iloc[0][col]
        vtype = type(val).__name__
        vstr = str(val)[:200] if not isinstance(val, (bytes, Image.Image, dict, list)) else f"[{vtype}, len={len(val) if hasattr(val, '__len__') else '?'}]"
        print(f"  {col}: {vtype} = {vstr}")

    os.makedirs(args.output, exist_ok=True)
    screenshot_dir = os.path.join(args.output, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    records = []
    skipped = 0

    for idx, row in df.iterrows():
        try:
            # --- instruction ---
            instruction = None
            for key in ["instruction", "goal", "task", "query", "prompt"]:
                if key in row and row[key]:
                    instruction = str(row[key])
                    break
            if not instruction:
                skipped += 1
                continue

            # --- image ---
            img = None
            for key in ["image", "screenshot", "img", "screen"]:
                if key in row:
                    img = extract_image(row[key])
                    if img:
                        break
            if img is None:
                skipped += 1
                continue

            img = img.convert("RGB")
            img = resize_if_needed(img)

            # --- task_id ---
            task_id = row.get("id", row.get("task_id", f"mobile_{idx}"))

            # --- Save screenshot ---
            img_filename = f"mobile_{task_id}_0.png"
            img.save(os.path.join(screenshot_dir, img_filename), "PNG")

            # --- JSONL record ---
            record = {
                "task_id": f"mobile_{task_id}",
                "instruction": instruction,
                "domain": "android",
                "traj": [
                    {"image": f"screenshots/{img_filename}"}
                ]
            }
            records.append(record)

            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(df)}...")

        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  Skipped row {idx}: {e}")

    # --- Write JSONL ---
    jsonl_path = os.path.join(args.output, "train.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n=== Results ===")
    print(f"Records: {len(records)} -> {jsonl_path}")
    print(f"Skipped: {skipped}")
    print(f"Screenshots: {screenshot_dir}")


if __name__ == "__main__":
    main()
