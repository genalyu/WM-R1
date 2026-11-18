#!/usr/bin/env bash
set -euo pipefail
cd /root/genalyu/WM-R1
mkdir -p logs
if ! curl --noproxy '*' -sf http://127.0.0.1:18890/health >/dev/null; then
 nohup bash scripts/serve_wm_fast.sh > "logs/wm-fast-$(date -u +%Y%m%dT%H%M%S).log" 2>&1 < /dev/null &
fi
if ! curl --noproxy '*' -sf http://127.0.0.1:18888/health >/dev/null; then
 nohup env PYTHONPATH=Code2World .venv/bin/python -u Code2World/android_world/agents/wm_vllm_proxy.py --speed-only > "logs/wm-proxy-$(date -u +%Y%m%dT%H%M%S).log" 2>&1 < /dev/null &
fi
export WM_EXTERNAL=1
exec bash scripts/run_pripro_subset.sh
