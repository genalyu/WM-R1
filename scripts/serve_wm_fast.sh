#!/usr/bin/env bash
set -euo pipefail
cd /root/genalyu/WM-R1
export CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=1
exec .venv-wm/bin/vllm serve models/Code2World --served-model-name code2world --host 127.0.0.1 --port 18890 --dtype bfloat16 --quantization fp8 --kv-cache-dtype auto --gpu-memory-utilization 0.93 --max-model-len 16384 --max-num-seqs 256 --max-num-batched-tokens 32768 --enable-prefix-caching --limit-mm-per-prompt '{"image":1,"video":0}' --mm-processor-kwargs '{"min_pixels":1024,"max_pixels":4096}'
