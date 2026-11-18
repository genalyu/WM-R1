"""Optional dynamic-batch WM service. Original serial server remains available."""
import argparse,base64,contextlib,io,json,queue,threading,time
from concurrent.futures import Future
from http.server import HTTPServer
from socketserver import ThreadingMixIn
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from android_world.agents.wm_server import WMHandler

def messages_for_processor(body):
    result=[]
    for msg in body.get("messages",[]):
        content=msg.get("content",[])
        if isinstance(content,list):
            converted=[]
            for item in content:
                if item.get("type")=="image_url":
                    url=item["image_url"]["url"]
                    if not url.startswith("data:"):raise ValueError("Only inline images are supported")
                    converted.append({"type":"image","image":Image.open(io.BytesIO(base64.b64decode(url.split(",",1)[1]))).convert("RGB")})
                else:converted.append(item)
            content=converted
        result.append({"role":msg.get("role","user"),"content":content})
    return result

class BatchEngine:
    def __init__(self,model,processor,batch_size,wait_ms):
        self.model=model;self.processor=processor
        self.batch_size=batch_size;self.wait_seconds=wait_ms/1000
        self.queue=queue.Queue()
        threading.Thread(target=self.loop,daemon=True).start()
    def submit(self,body):
        future=Future();self.queue.put((body,future));return future.result(timeout=900)
    def loop(self):
        while True:
            first=self.queue.get()
            batch=[first]
            key=lambda body:(body.get("max_tokens",2048),body.get("temperature",0.0))
            deadline=time.monotonic()+self.wait_seconds
            while len(batch)<self.batch_size:
                remaining=deadline-time.monotonic()
                try:item=self.queue.get(timeout=max(remaining,0))
                except queue.Empty:break
                if key(item[0])!=key(first[0]):
                    self.queue.put(item);break
                batch.append(item)
            started=time.monotonic()
            try:
                conversations=[messages_for_processor(b) for b,_ in batch]
                inputs=self.processor.apply_chat_template(conversations,add_generation_prompt=True,tokenize=True,padding=True,return_dict=True,return_tensors="pt").to(self.model.device)
                max_tokens,temp=key(first[0])
                kwargs={"max_new_tokens":max_tokens,"do_sample":temp>0,"use_cache":True}
                if temp>0:kwargs["temperature"]=temp
                with torch.inference_mode():outputs=self.model.generate(**inputs,**kwargs)
                trimmed=outputs[:,inputs.input_ids.shape[1]:]
                texts=self.processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                ended=time.monotonic()
                pad=self.processor.tokenizer.pad_token_id
                counts=[int((row!=pad).sum().item()) for row in trimmed]
                for (_,future),text,n_tokens in zip(batch,texts,counts):
                    future.set_result((text,n_tokens,len(batch),started,ended))
                print(json.dumps({"event":"wm_batch","batch_size":len(batch),"seconds":ended-started,"output_tokens":sum(counts)}),flush=True)
                del inputs,outputs,trimmed
            except Exception as exc:
                for _,future in batch:future.set_exception(exc)

class BatchWMHandler(WMHandler):
    _lock=contextlib.nullcontext()
    engine=None
    def infer_from_openai_messages(self,body):
        text,count,size,started,ended=self.engine.submit(body)
        self.last_generation_tokens=count
        self.last_batch_size=size
        self.batch_started_at=started
        self.batch_finished_at=ended
        return text

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--port",type=int,default=18888)
    parser.add_argument("--max-batch-size",type=int,default=4)
    parser.add_argument("--batch-wait-ms",type=float,default=30)
    args=parser.parse_args()
    processor=AutoProcessor.from_pretrained(args.model)
    processor.tokenizer.padding_side="left"
    model=Qwen3VLForConditionalGeneration.from_pretrained(args.model,device_map=args.device,torch_dtype=torch.bfloat16,attn_implementation="sdpa").eval()
    BatchWMHandler.engine=BatchEngine(model,processor,args.max_batch_size,args.batch_wait_ms)
    server=type("BatchedHTTPServer",(ThreadingMixIn,HTTPServer),{"daemon_threads":True})(("127.0.0.1",args.port),BatchWMHandler)
    print("Batched WM server ready; max_batch_size=",args.max_batch_size,flush=True)
    server.serve_forever()

