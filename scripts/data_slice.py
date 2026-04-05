import json
import random
import os

# 定义输入和输出路径
input_file = '/public/home/xlwang/genalyu/dataset/agentnet/agentnet_10k.jsonl'
train_file = '/public/home/xlwang/genalyu/dataset/agentnet/agentnet_train_9k.jsonl'
test_file = '/public/home/xlwang/genalyu/dataset/agentnet/agentnet_test_1k.jsonl'

# 设定训练集比例 (90% 训练, 10% 测试)
train_ratio = 0.9

def split_dataset(input_path, train_path, test_path, ratio):
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 {input_path}")
        return

    # 1. 读取所有数据
    print(f"正在读取数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]

    total_count = len(lines)
    print(f"总计条数: {total_count}")

    # 2. 随机打乱顺序
    random.shuffle(lines)

    # 3. 计算切分点
    split_index = int(total_count * ratio)
    train_data = lines[:split_index]
    test_data = lines[split_index:]

    # 4. 保存文件
    print(f"正在保存训练集 ({len(train_data)} 条) -> {train_path}")
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_data)

    print(f"正在保存测试集 ({len(test_data)} 条) -> {test_path}")
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(test_data)

    print("数据集切分完成！")

if __name__ == "__main__":
    # 你可以根据需要调整 ratio，比如 0.8 表示 8:2
    split_dataset(input_file, train_file, test_file, train_ratio)