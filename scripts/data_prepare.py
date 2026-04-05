import json
import random

# 定义输入文件路径
file1 = '/public/home/xlwang/genalyu/dataset/agentnet/agentnet_ubuntu_5k.jsonl'
file2 = '/public/home/xlwang/genalyu/dataset/agentnet/agentnet_win_mac_18k.jsonl'
output_file1 = '/public/home/xlwang/genalyu/dataset/agentnet/agentnet_full.jsonl'
output_file2 = '/public/home/xlwang/genalyu/dataset/agentnet/agentnet_10k.jsonl'

def merge_and_sample(f1, f2, out1, out2, sample_size=10000):
    all_data = []

    print(f"正在读取 {f1}...")
    with open(f1, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(line)
                
    print(f"正在读取 {f2}...")
    with open(f2, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_data.append(line)
    
    total_count = len(all_data)
    print(f"合并完成，总行数: {total_count}")
    
    print(f"正在保存至 {out1}...")
    with open(out1, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(item)
    # 随机采样
    if total_count <= sample_size:
        print(f"总数据量不足 {sample_size}，将保留全部数据。")
        sampled_data = all_data
    else:
        print(f"正在随机抽取 {sample_size} 条...")
        sampled_data = random.sample(all_data, sample_size)
    
    # 写入新文件
    print(f"正在保存至 {out2}...")
    with open(out2, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(item)
            
    print("任务完成！")

if __name__ == "__main__":
    merge_and_sample(file1, file2, output_file1, output_file2)