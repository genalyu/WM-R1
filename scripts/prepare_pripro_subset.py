from pathlib import Path
import pandas as pd, json,random,hashlib,io
from PIL import Image
ROOT=Path("/root/genalyu/WM-R1")
source=ROOT/"data/raw/gui_r1_source_groups17_18.parquet"
out=ROOT/"data/guir1_subset";(out/"screenshots").mkdir(parents=True,exist_ok=True)
df=pd.read_parquet(source);print("source rows",len(df),"columns",list(df.columns),flush=True)
order=list(range(len(df)));random.Random(1).shuffle(order)
records=[];seen=set();manifest=[]
for idx in order:
 row=df.iloc[idx];value=row["image"]
 if isinstance(value,dict):value=value.get("bytes")
 if value is None:continue
 image=Image.open(io.BytesIO(value));w,h=image.size
 if h<1.4*w:continue
 digest=hashlib.sha256(value).hexdigest()
 if digest in seen:continue
 seen.add(digest)
 instruction=str(row["instruction"]).strip()
 if not instruction:continue
 name=f"guir1_{idx}.png"
 image.convert("RGB").resize((1080,2400),Image.Resampling.LANCZOS).save(out/"screenshots"/name)
 records.append({"task_id":f"guir1_train_{idx}","instruction":instruction,"domain":"android","traj":[{"image":"screenshots/"+name}]})
 manifest.append({"source_index":int(row.get("source_index",idx)),"source_id":str(row.get("id","")),"original_size":[w,h],"original_image_sha256":digest,"task_type":str(row.get("task_type",""))})
 if len(records)==40:break
assert len(records)==40,f"Only {len(records)} portrait samples available"
for split,rows in [("train",records[:32]),("val",records[32:])]:
 (out/(split+".jsonl")).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
(out/"manifest.json").write_text(json.dumps({"dataset":"ritzzai/GUI-R1","source_file":"train.parquet","source_row_groups":[17,18],"extracted_source_file":source.name,"extracted_source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"source_expected_sha256":"cd6ca00dfabafe77ada53a12d5103b5e93a27b9891046862e736436baf1efe09","selection":"row groups 17 and 18; seed=1 shuffled indices; portrait h>=1.4*w; unique image SHA256; first 32 train, next 8 validation","note":"Small pipeline pilot; not the paper's curated 2000-task set or an official evaluation split.","samples":manifest},indent=2))
print("Created 32 train + 8 validation records with disjoint screenshot hashes",flush=True)

