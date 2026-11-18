import sys,glob,json
from wandb.sdk.internal.datastore import DataStore
from wandb.proto.wandb_internal_pb2 import Record
for f in glob.glob(sys.argv[1]+"/*/wandb/offline*/run*.wandb"):
 print(f)
 d=DataStore();d.open_for_scan(f)
 while (v:=d.scan_data()):
  r=Record();r.ParseFromString(v)
  if r.HasField("history"):
   h={i.key or ".".join(i.nested_key):i.value_json for i in r.history.item}
   print({k:v for k,v in h.items() if k in ("_step","throughput/completed_env_steps_per_second","wm/transition_requests","throughput/completed_env_steps","actor/grad_norm","wm/train_iteration_seconds")})
