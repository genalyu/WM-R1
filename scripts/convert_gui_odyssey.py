#!/usr/bin/env python3
"""Convert GUI-Odyssey dataset to WM-R1 unified JSONL format.

Usage:
    python scripts/convert_gui_odyssey.py \
        --input_dir /path/to/GUI-Odyssey/ \
        --output data/gui_odyssey/

    # Or from HuggingFace:
    python scripts/convert_gui_odyssey.py \
        --source hf \
        --output data/gui_odyssey/

Output:
    data/gui_odyssey/train.jsonl
    data/gui_odyssey/screenshots/*.png
"""
import argparse
import json
import os
import sys
import glob
from io import BytesIO
from PIL import Image


TARGET_WIDTH = 1080
TARGET_HEIGHT = 2400


def resize_if_needed(img: Image.Image) -> Image.Image:
    """Resize image to target dimensions if needed."""
    if img.size == (TARGET_WIDTH, TARGET_HEIGHT):
        return img
    return img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)


def convert_from_local(input_dir: str, output_dir: str):
    """Convert from local GUI-Odyssey directory."""
    screenshot_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    # Find annotation files
    # GUI-Odyssey stores annotations as JSON files, one per device or split
    annotation_files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)
    if not annotation_files:
        # Try common locations
        for pattern in ["annotations/*.json", "data/*.json", "*.json"]:
            annotation_files.extend(glob.glob(os.path.join(input_dir, pattern)))

    if not annotation_files:
        print(f"No JSON annotation files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(annotation_files)} annotation files")

    records = []
    skipped = 0

    for ann_file in annotation_files:
        print(f"Processing {os.path.basename(ann_file)}...")
        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # GUI-Odyssey format: list of episodes
        episodes = data if isinstance(data, list) else data.get("episodes", data.get("data", []))

        for ep_idx, episode in enumerate(episodes):
            try:
                # Extract episode info
                episode_id = episode.get("episode_id", episode.get("id", f"ep_{ep_idx}"))
                instruction = episode.get("instruction", episode.get("goal", episode.get("task", "")))

                if not instruction:
                    skipped += 1
                    continue

                # Get first screenshot
                steps = episode.get("steps", episode.get("trajectory", episode.get("actions", [])))
                if not steps:
                    skipped += 1
                    continue

                first_step = steps[0]
                img_path = first_step.get("screenshot", first_step.get("image", first_step.get("img_path", "")))

                if not img_path:
                    skipped += 1
                    continue

                # Resolve image path (may be relative to input_dir)
                if not os.path.isabs(img_path):
                    img_path = os.path.join(input_dir, img_path)

                if not os.path.exists(img_path):
                    # Try common image directories
                    img_name = os.path.basename(img_path)
                    for search_dir in ["images", "screenshots", "imgs", "data"]:
                        candidate = os.path.join(input_dir, search_dir, img_name)
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
                    else:
                        skipped += 1
                        continue

                # Load and resize image
                img = Image.open(img_path).convert("RGB")
                img = resize_if_needed(img)

                # Save to output directory
                img_filename = f"go_{episode_id}_0.png"
                out_img_path = os.path.join(screenshot_dir, img_filename)
                img.save(out_img_path, "PNG")

                # Create JSONL record
                record = {
                    "task_id": f"gui_odyssey_{episode_id}",
                    "instruction": instruction,
                    "domain": "android",
                    "traj": [
                        {"image": f"screenshots/{img_filename}"}
                    ]
                }
                records.append(record)

            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"  Skipped episode {ep_idx}: {e}")

    # Write JSONL
    jsonl_path = os.path.join(output_dir, "train.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(records)} records → {jsonl_path}")
    print(f"Skipped: {skipped}")


def convert_from_huggingface(output_dir: str, dataset_name: str = "OpenGVLab/GUI-Odyssey"):
    """Convert from HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install datasets: pip install datasets")
        sys.exit(1)

    print(f"Loading {dataset_name} from HuggingFace...")
    ds = load_dataset(dataset_name, split="train")
    print(f"Loaded {len(ds)} examples")

    screenshot_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    records = []
    skipped = 0

    for idx, example in enumerate(ds):
        try:
            episode_id = example.get("episode_id", example.get("id", str(idx)))
            instruction = example.get("instruction", example.get("goal", ""))

            if not instruction:
                skipped += 1
                continue

            # Try to get first screenshot from various possible formats
            img = None

            # Format 1: screenshots list
            screenshots = example.get("screenshots", [])
            if screenshots:
                img_data = screenshots[0]
                if isinstance(img_data, Image.Image):
                    img = img_data
                elif isinstance(img_data, dict) and "bytes" in img_data:
                    img = Image.open(BytesIO(img_data["bytes"]))
                elif isinstance(img_data, bytes):
                    img = Image.open(BytesIO(img_data))

            # Format 2: image field
            if img is None:
                img_field = example.get("image", None)
                if isinstance(img_field, Image.Image):
                    img = img_field
                elif isinstance(img_field, dict) and "bytes" in img_field:
                    img = Image.open(BytesIO(img_field["bytes"]))

            if img is None:
                skipped += 1
                continue

            img = img.convert("RGB")
            img = resize_if_needed(img)

            img_filename = f"go_{episode_id}_0.png"
            img.save(os.path.join(screenshot_dir, img_filename), "PNG")

            record = {
                "task_id": f"gui_odyssey_{episode_id}",
                "instruction": instruction,
                "domain": "android",
                "traj": [
                    {"image": f"screenshots/{img_filename}"}
                ]
            }
            records.append(record)

            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(ds)}...")

        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  Skipped example {idx}: {e}")

    jsonl_path = os.path.join(output_dir, "train.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(records)} records → {jsonl_path}")
    print(f"Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Convert GUI-Odyssey to WM-R1 JSONL")
    parser.add_argument("--source", choices=["local", "hf"], default="hf",
                        help="Data source: 'local' or 'hf' (HuggingFace)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory (for local source)")
    parser.add_argument("--output", type=str, default="data/gui_odyssey",
                        help="Output directory")
    parser.add_argument("--dataset", type=str, default="OpenGVLab/GUI-Odyssey",
                        help="HuggingFace dataset name")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.source == "hf":
        convert_from_huggingface(args.output, args.dataset)
    else:
        if not args.input_dir:
            print("ERROR: --input_dir required for local source")
            sys.exit(1)
        convert_from_local(args.input_dir, args.output)


if __name__ == "__main__":
    main()
