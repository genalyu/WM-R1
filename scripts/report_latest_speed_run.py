import os,json
from pathlib import Path
import wandb
root=Path("/root/genalyu/WM-R1")
os.environ["WANDB_API_KEY"]=(root/".secrets/wandb.key").read_text().strip()
os.environ["WANDB_BASE_URL"]="https://api.wandb.ai"
run_dir=Path((root/"runs/latest-run.txt").read_text().strip())
local_runs=sorted((run_dir/"wandb").glob("run-*"))
run_id=local_runs[-1].name.rsplit("-",1)[1]
run=wandb.Api(timeout=60).run("genal-yu/wm-r1-android/"+run_id)
keys=["_step","throughput/completed_env_steps","throughput/completed_env_steps_per_second","throughput/cumulative_env_steps_per_second","wm/train_iteration_seconds","timing_s/update_actor","actor/grad_norm"]
history=list(run.scan_history(keys=keys))
result={"url":run.url,"state":run.state,"run_dir":str(run_dir),"history":history,"profile":json.loads((root/"logs/speed-profile.json").read_text())}
(root/"logs/latest-speed-result.json").write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!="profile"},indent=2))
