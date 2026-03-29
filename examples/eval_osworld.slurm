#!/bin/bash
#SBATCH --job-name=osworld_eval
#SBATCH --partition=a100          # 根据你的分区名修改
#SBATCH --nodes=1                 # 4 个节点即可
#SBATCH --gres=gpu:2              # 评估通常需要多张卡来跑 vLLM
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=256G
#SBATCH --output=/public/home/xlwang/genalyu/WM-R1/logs/eval_%j.out
#SBATCH --error=/public/home/xlwang/genalyu/WM-R1/logs/eval_%j.err

source ~/.bashrc
# --- 关键：环境初始化 (Conda 激活) ---
# 1. 显式加载 Conda 的基础配置
source /public/home/xlwang/jyy/anaconda/etc/profile.d/conda.sh  # 路径根据你安装 anaconda/miniconda 的位置修改
# 或者使用: source $(conda info --base)/etc/profile.d/conda.sh

# 2. 激活环境
conda activate wm-r1

# 3. 确保当前目录在项目根目录
cd /public/home/xlwang/genalyu/WM-R1  # 替换为你的真实绝对路径

# --- 步骤 1: 启动模型推理服务 ---
echo "Starting model server..."
# 注意：在 Slurm 内部不需要 nohup，因为整个脚本就是在后台运行的
# 但为了让它在后台跑，我们用 & 并记录 PID
bash start_server.sh > logs/vllm_debug_${SLURM_JOB_ID}.log 2>&1 &
SERVER_PID=$!
echo "Waiting for vLLM to be ready..."
for i in {1..50}; do
    # 假设你的 vLLM 跑在 9000 端口
    if curl -s http://127.0.0.1:9000/v1/models > /dev/null; then
        echo "vLLM is READY!"
        break
    fi
    echo "Wait $i: Server not ready yet..."
    sleep 10
done
# --- 步骤 2: 运行 OSWorld 评估脚本 ---
echo "Starting OSWorld evaluation..."
cd OSWorld

python run_multienv_uitars.py \
    --headless \
    --observation_type screenshot \
    --max_steps 15 \
    --max_trajectory_length 15 \
    --temperature 0.6 \
    --model evocua-8b\
    --action_space pyautogui \
    --num_envs 8 \
    --result_dir ./results/ \
    --test_all_meta_path ./evaluation_examples/test_all.json \
    --trial-id 0 \
    --path_to_vm /public/home/xlwang/genalyu/WM-R1/OSWorld/docker_vm_data/Ubuntu.qcow2 \
    --server_ip http://127.0.0.1

# --- 步骤 3: 清理 ---
echo "Evaluation finished. Killing model server..."
kill $SERVER_PID