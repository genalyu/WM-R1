#!/bin/bash
set -eu
d=/home/tione/notebook/genalyu/training-2gpu-20261118
p=/root/genalyu/WM-R1
mkdir -p "$p/logs" "$d/results" /root/.cache
exec > >(tee -a "$d/attempt2/cloud.log") 2>&1
date -Is
cd "$p"
tar xf "$d/source.tar"
tar xf "$d/runtime.tar"
tar xf "$d/browser.tar" -C /root/.cache
ln -s "$d/models" models
mkdir -p runs
mkdir -p "$d/attempt2/results"
trap 'cp -r runs/* "$d/attempt2/results/"' EXIT
python3 - <<'PY'
from pathlib import Path
r=Path("verl/trainer/ray_trainer.py")
t=r.read_text()
a=t.index("        self.data_processor_workers = [")
b=t.index("        ]",a)+len("        ]")
assert t.count("self.data_processor_workers")==1
r.write_text(t[:a]+"        self.data_processor_workers = []"+t[b:])
p=Path("scripts/run_pripro_subset.sh")
s=p.read_text()
a=s.index("export WANDB_API_KEY"); b=s.index("export WANDB_ENTITY",a)
s=s[:a]+s[b:]
s=s.replace("export WANDB_MODE=online","export WANDB_MODE=offline")
s="\n".join(x for x in s.split("\n") if not x.startswith(("export HTTPS_PROXY","export HTTP_PROXY")))
a=s.index('"$PYTHON" - <<\'CHECK\''); b=s.index('\nCHECK\n',a)
p.write_text(s[:a]+s[b+7:])
PY
python3 - <<'PATCH'
from pathlib import Path
p=Path("Code2World/android_world/agents/wm_utils.py")
s=p.read_text().replace('return _render_pool.submit(_render_fast, html_path, output_path).result(timeout=60)','import json, urllib.request\n        req=urllib.request.Request("http://127.0.0.1:18901",data=json.dumps([str(html_path),str(output_path)]).encode())\n        with urllib.request.urlopen(req,timeout=70) as response: return response.status==200')
s=s.replace('viewport={"width":1080,"height":2400}', 'viewport={"width":108,"height":240}')
p.write_text(s)
PATCH
export PYTHONPATH="$p:$p/Code2World"
.venv/bin/python "$d/attempt2/render.py" >logs/render.log 2>&1 &
export CUDA_VISIBLE_DEVICES=1
.venv/bin/ray start --head --port=6379 --num-gpus=1 --num-cpus=48 --object-store-memory=4000000000
bash scripts/run_speed_profile.sh
