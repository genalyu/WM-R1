import json
import os
import random

# --- 配置 ---
data_dir = "/public/home/xlwang/genalyu/dataset/agentnet"

# 输入 JSONL 文件
input_files = [
    "agentnet_ubuntu_5k.jsonl",
    "agentnet_win_mac_18k.jsonl",
]

# 输出文件
output_train = "agentnet_train_9k.jsonl"
output_test = "agentnet_test_1k.jsonl"
output_full = "agentnet_full.jsonl"

# 验证集大小
test_size = 100

# 图片目录
images_dir = os.path.join(data_dir, "images")

def verify_image_in_images_dir(item):
    """验证 traj[0] 的图片能在 images 目录下找到"""
    traj = item.get("traj", [])
    if not traj:
        return False
    image_filename = traj[0].get("image", "")
    if not image_filename or not isinstance(image_filename, str):
        return False
    # 只取文件名，去除可能的路径前缀
    image_filename = os.path.basename(image_filename)
    image_path = os.path.join(images_dir, image_filename)
    return os.path.exists(image_path)

def main():
    all_data = []
    skipped_count = 0

    for filename in input_files:
        filepath = os.path.join(data_dir, filename)
        print(f"读取 {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if verify_image_in_images_dir(item):
                    all_data.append(item)
                else:
                    traj = item.get("traj", [])
                    image_name = traj[0].get("image", "N/A") if traj else "N/A"
                    skipped_count += 1
                    if skipped_count <= 10:
                        print(f"  跳过 (图片不存在): {item.get('task_id', 'unknown')} -> {image_name}")

    print(f"\n合并完成，有效数据: {len(all_data)} 条")
    print(f"跳过 (图片缺失): {skipped_count} 条")

    # 保存完整合并文件
    output_full_path = os.path.join(data_dir, output_full)
    print(f"保存至 {output_full_path}...")
    with open(output_full_path, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 随机打乱并划分训练集/验证集
    random.seed(42)
    random.shuffle(all_data)

    test_data = all_data[:test_size]
    train_data = all_data[test_size:]

    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(test_data)} 条")

    # 保存训练集
    train_path = os.path.join(data_dir, output_train)
    print(f"保存至 {train_path}...")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 保存验证集
    test_path = os.path.join(data_dir, output_test)
    print(f"保存至 {test_path}...")
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n任务完成！")

if __name__ == "__main__":
    main()
