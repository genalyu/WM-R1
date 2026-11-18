"""OpenAI-compatible WM proxy with metrics and optional deterministic exact-request cache."""
import argparse,contextlib,hashlib,json,sqlite3,threading,time
from concurrent.futures import Future
from http.server import HTTPServer
from socketserver import ThreadingMixIn
from pathlib import Path
import requests
from android_world.agents.wm_server import WMHandler

class ProxyWMHandler(WMHandler):
    _lock=contextlib.nullcontext()
    backend="http://127.0.0.1:18890/v1/chat/completions"
    speed_only=False
    cache=None
    cache_lock=threading.Lock()
    inflight={}
    local=threading.local()
    def infer_from_openai_messages(self,body):
        payload={k:v for k,v in body.items() if not k.startswith("wm_")}
        payload["model"]="code2world"
        payload["chat_template_kwargs"]={"enable_thinking":False}
        payload.setdefault("temperature",0.0)
        if self.speed_only:
            payload.pop("structured_outputs",None)
            payload["max_tokens"]=96 if body.get("wm_request_kind") == "judge" else 64
        key=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        cacheable=self.cache is not None and payload["temperature"]==0
        owner=True;future=None;hit=False
        if cacheable:
            with self.cache_lock:
                row=self.cache.execute("SELECT value FROM responses WHERE key=?",(key,)).fetchone()
                if row:
                    result=json.loads(row[0]);hit=True
                else:
                    future=self.inflight.get(key)
                    if future is None:
                        future=Future();self.inflight[key]=future
                    else:owner=False
            if not hit and not owner:
                result=future.result(timeout=900);hit=True
        if not hit:
            try:
                if not hasattr(self.local,"session"):
                    self.local.session=requests.Session()
                    self.local.session.trust_env=False
                r=self.local.session.post(self.backend,json=payload,timeout=900)
                r.raise_for_status()
                result=r.json()
                choice=result["choices"][0]
                if not choice["message"].get("content"):raise ValueError("Empty WM response")
                if cacheable:
                    with self.cache_lock:
                        self.cache.execute("INSERT OR REPLACE INTO responses VALUES (?,?)",(key,json.dumps(result)))
                        self.cache.commit();self.inflight.pop(key,None)
                    future.set_result(result)
                with self._metrics_lock:
                    self.metrics["backend_requests"]=self.metrics.get("backend_requests",0)+1
                    self.metrics["backend_generated_tokens"]=self.metrics.get("backend_generated_tokens",0)+result.get("usage",{}).get("completion_tokens",0)
            except Exception as exc:
                if cacheable and future is not None:
                    with self.cache_lock:self.inflight.pop(key,None)
                    future.set_exception(exc)
                raise
        if hit:
            with self._metrics_lock:self.metrics["exact_cache_hits"]=self.metrics.get("exact_cache_hits",0)+1
        self.last_generation_tokens=result.get("usage",{}).get("completion_tokens",0)
        self.last_batch_size=None  # continuous batching is tracked by the vLLM /metrics endpoint
        self.last_cache_hit=hit
        finish=result["choices"][0].get("finish_reason")
        if finish=="length":
            with self._metrics_lock:self.metrics["length_truncations"]=self.metrics.get("length_truncations",0)+1
        return result["choices"][0]["message"]["content"]

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--port",type=int,default=18888)
    p.add_argument("--backend",default="http://127.0.0.1:18890/v1/chat/completions")
    p.add_argument("--cache-dir")
    p.add_argument("--speed-only",action="store_true")
    a=p.parse_args()
    ProxyWMHandler.backend=a.backend
    ProxyWMHandler.speed_only=a.speed_only
    if a.cache_dir:
        path=Path(a.cache_dir);path.mkdir(parents=True,exist_ok=True)
        ProxyWMHandler.cache=sqlite3.connect(path/"responses.sqlite",check_same_thread=False)
        ProxyWMHandler.cache.execute("PRAGMA journal_mode=WAL")
        ProxyWMHandler.cache.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        ProxyWMHandler.cache.commit()
    server=type("ProxyHTTPServer",(ThreadingMixIn,HTTPServer),{"daemon_threads":True,"request_queue_size":1024})(("127.0.0.1",a.port),ProxyWMHandler)
    print("WM proxy ready; exact cache enabled:",bool(a.cache_dir),flush=True)
    server.serve_forever()

