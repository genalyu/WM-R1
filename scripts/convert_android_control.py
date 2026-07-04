#!/usr/bin/env python3
"""Convert AndroidControl dataset to WM-R1 unified JSONL format.

Usage:
    # From HuggingFace Parquet (recommended):
    python scripts/convert_android_control.py \
        --source hf \
        --output data/android_control/

    # From local TFRecord:
    python scripts/convert_android_control.py \
        --source tfrecord \
        --input_dir /path/to/android_control/ \
        --output data/android_control/

Output:
    data/android_control/train.jsonl
    data/android_control/screenshots/*.png
"""
import argparse
import json
import os
import sys
from io import BytesIO
from PIL import Image


TARGET_WIDTH = 1080
TARGET_HEIGHT = 2400


def convert_from_huggingface(output_dir: str, dataset_name: str = "InfiX-ai/android_control_train"):
    """Convert from HuggingFace Parquet dataset."""
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
            # Extract fields - HF processed format
            episode_id = example.get("episode_id", str(idx))
            goal = example.get("goal", example.get("instruction", ""))

            if not goal:
                skipped += 1
                continue

            # Get first screenshot
            screenshots = example.get("screenshots", [])
            if not screenshots:
                skipped += 1
                continue

            # Save first screenshot
            img_data = screenshots[0]
            if isinstance(img_data, dict) and "bytes" in img_data:
                img_bytes = img_data["bytes"]
            elif isinstance(img_data, bytes):
                img_bytes = img_data
            elif isinstance(img_data, Image.Image):
                buf = BytesIO()
                img_data.save(buf, format="PNG")
                img_bytes = buf.getvalue()
            else:
                skipped += 1
                continue

            # Verify image is valid
            img = Image.open(BytesIO(img_bytes))
            img.verify()

            # Save screenshot
            img_filename = f"ac_{episode_id}_0.png"
            img_path = os.path.join(screenshot_dir, img_filename)
            with open(img_path, "wb") as f:
                f.write(img_bytes if isinstance(img_bytes, bytes) else img_bytes)

            # Create JSONL record
            record = {
                "task_id": f"android_control_{episode_id}",
                "instruction": goal,
                "domain": "android",
                "traj": [
                    {"image": f"screenshots/{img_filename}"}
                ]
            }
            records.append(record)

            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(ds)}...")

        except Exception as e:
            skipped += 1
            if skipped <= 10:
                print(f"  Skipped example {idx}: {e}")

    # Write JSONL
    jsonl_path = os.path.join(output_dir, "train.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(records)} records → {jsonl_path}")
    print(f"Skipped: {skipped}")
    print(f"Screenshots: {screenshot_dir}")


def convert_from_tfrecord(input_dir: str, output_dir: str):
    """Convert from raw TFRecord files."""
    try:
        import tensorflow as tf
    except ImportError:
        print("ERROR: Install tensorflow: pip install tensorflow")
        sys.exit(1)

    screenshot_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    # Find TFRecord files
    tfrecord_files = []
    for f in os.listdir(input_dir):
        if f.endswith(".tfrecord") or f.endswith(".tfrecord.gz"):
            tfrecord_files.append(os.path.join(input_dir, f))

    if not tfrecord_files:
        print(f"No TFRecord files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(tfrecord_files)} TFRecord files")

    records = []
    skipped = 0

    for tfrecord_path in tfrecord_files:
        print(f"Processing {os.path.basename(tfrecord_path)}...")

        raw_dataset = tf.data.TFRecordDataset(
            tfrecord_path,
            compression_type="GZIP" if tfrecord_path.endswith(".gz") else None
        )

        for raw_record in raw_dataset:
            try:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())

                feat = example.features.feature

                episode_id = feat["episode_id"].int64_list.value[0]
                goal = feat["goal"].bytes_list.value[0].decode("utf-8")

                screenshots = feat["screenshots"].bytes_list.value
                if not screenshots:
                    skipped += 1
                    continue

                # Save first screenshot
                img_filename = f"ac_{episode_id}_0.png"
                img_path = os.path.join(screenshot_dir, img_filename)
                with open(img_path, "wb") as f:
                    f.write(screenshots[0])

                record = {
                    "task_id": f"android_control_{episode_id}",
                    "instruction": goal,
                    "domain": "android",
                    "traj": [
                        {"image": f"screenshots/{img_filename}"}
                    ]
                }
                records.append(record)

            except Exception as e:
                skipped += 1
                if skipped <= 10:
                    print(f"  Skipped: {e}")

    # Write JSONL
    jsonl_path = os.path.join(output_dir, "train.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nDone! {len(records)} records → {jsonl_path}")
    print(f"Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Convert AndroidControl to WM-R1 JSONL")
    parser.add_argument("--source", choices=["hf", "tfrecord"], default="hf",
                        help="Data source: 'hf' (HuggingFace Parquet) or 'tfrecord'")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Input directory (for tfrecord source)")
    parser.add_argument("--output", type=str, default="data/android_control",
                        help="Output directory")
    parser.add_argument("--dataset", type=str, default="InfiX-ai/android_control_train",
                        help="HuggingFace dataset name")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.source == "hf":
        convert_from_huggingface(args.output, args.dataset)
    else:
        if not args.input_dir:
            print("ERROR: --input_dir required for tfrecord source")
            sys.exit(1)
        convert_from_tfrecord(args.input_dir, args.output)


if __name__ == "__main__":
    main()
