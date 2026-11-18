import argparse,time,json,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from PIL import Image
from verl.trainer.gui_agent import QwenHTTPClient
p=argparse.ArgumentParser();p.add_argument("--concurrency",type=int,default=4);p.add_argument("--requests",type=int,default=4);p.add_argument("--tag",default="bf16");p.add_argument("--dataset",action="store_true");a=p.parse_args()
root=Path("/root/genalyu/WM-R1");out=root/"runs"/("wm-probe-"+a.tag+"-"+time.strftime("%Y%m%dT%H%M%S"));out.mkdir(parents=True)
im=Image.open(root/"logs/demo_case.png").convert("RGB")
rows=[json.loads(x) for x in (root/"data/guir1_subset/train.jsonl").read_text().splitlines()] if a.dataset else []
def one(i):
 client=QwenHTTPClient("http://127.0.0.1:18888",str(out/f"request_{i}"))
 t=time.perf_counter()
 sample=rows[i%len(rows)] if rows else None
 screen=Image.open(root/"data/guir1_subset"/sample["traj"][0]["image"]).convert("RGB") if sample else im.copy()
 arr=client.post(sample["instruction"] if sample else "Change the number of adult passengers","Tap selected location",screen,f"tap(x={100+i%900}, y=1290)",idx=i)
 return {"i":i,"seconds":time.perf_counter()-t,"shape":list(arr.shape)}
t=time.perf_counter();results=[];errors=[]
with ThreadPoolExecutor(max_workers=a.concurrency) as pool:
 for f in as_completed([pool.submit(one,i) for i in range(a.requests)]):
  try:
   r=f.result();results.append(r);print("FINISHED",json.dumps(r),flush=True)
  except Exception as e:
   errors.append(str(e));print("FAILED",str(e),flush=True)
elapsed=time.perf_counter()-t
summary={"tag":a.tag,"errors":errors,"successful":len(results),"concurrency":a.concurrency,"requests":a.requests,"seconds":elapsed,"wm_and_render_requests_per_second":len(results)/elapsed,"dataset":bool(a.dataset),"scope":"component-only; no policy generation or training updates; not end-to-end training throughput","profile":{"speed_only":os.getenv("WM_SPEED_ONLY"),"max_output_tokens":os.getenv("WM_MAX_OUTPUT_TOKENS"),"accepts_truncated_html":True},"results":results}
(out/"summary.json").write_text(json.dumps(summary,indent=2))
print("SUMMARY",json.dumps(summary),flush=True)

