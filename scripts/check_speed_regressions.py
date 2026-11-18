"""CPU regression checks for the throughput fixes, not throughput measurements."""
from types import SimpleNamespace
import math
import torch
import torch.nn.functional as F
from verl.utils.torch_functional import postprocess_data, log_probs_from_logits
from verl.trainer.metrics import compute_data_metrics

ids=torch.tensor([1,9,9,2,9,9,3])
grids=torch.tensor([[1,2,4],[1,2,4]])
pixels=torch.arange(16).reshape(16,1)
for direction,limit,expected in [("right",5,8),("left",5,8),("right",3,8),("right",2,0),("left",2,0),("right",10,16)]:
 x,m,pos,lab,pv,g=postprocess_data(ids,torch.ones_like(ids),torch.arange(7).repeat(3,1),limit,0,truncation=direction,labels=ids.clone(),image_token_id=9,pixel_values=pixels,image_grid_thw=grids)
 assert len(x)==len(m)==len(lab)==pos.shape[-1]==limit
 assert len(pv)==expected
 assert int((x==9).sum())==sum(int(t)*(int(h)//2)*(int(w)//2) for t,h,w in g)
 if direction=="left" and limit==5:assert torch.equal(pv,pixels[8:])
torch.manual_seed(1)
x=torch.randn(3,7,101,requires_grad=True);y=x.detach().clone().requires_grad_(True)
labels=torch.randint(101,(3,7));mask=torch.rand(3,7)>.6;weights=torch.randn(3,7)
a=log_probs_from_logits(x,labels)
expected=torch.log_softmax(x,-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
assert torch.allclose(a,expected,atol=1e-6)
sel=log_probs_from_logits(y[mask],labels[mask])
assert sel.ndim==1
b=torch.zeros(3,7);b[mask]=sel
(a*mask*weights).sum().backward();(b*mask*weights).sum().backward()
assert torch.allclose(a[mask],b[mask],atol=1e-6)
assert torch.allclose(x.grad,y.grad,atol=1e-6)
d={"input_ids":torch.tensor([[0,1,2,3,4],[0,1,2,5,6]]),"labels":torch.tensor([[-100,-100,-100,3,4],[-100,-100,-100,5,6]]),"attention_mask":torch.tensor([[0,1,1,1,1],[0,1,1,1,1]]),"token_level_scores":torch.ones(2,1),"token_level_rewards":torch.ones(2,1),"advantages":torch.ones(2,4),"returns":torch.ones(2,4),"traj_len":torch.ones(2,1)}
d["responses"]=d["input_ids"][:,1:];d["response_mask"]=d["labels"][:,1:]!=-100
metrics=compute_data_metrics(SimpleNamespace(batch=d))
assert all(math.isfinite(x) for x in metrics.values())
assert metrics["response_length/mean"]==2


# Exercise the actual actor forward path, including its selected-token indexing.
import os,copy
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.actor.config import ActorConfig
class TinyLM(torch.nn.Module):
 def __init__(self):
  super().__init__();self.embedding=torch.nn.Embedding(101,8);self.head=torch.nn.Linear(8,101)
 def forward(self,input_ids,**kwargs):
  return SimpleNamespace(logits=self.head(self.embedding(input_ids)))
model=TinyLM();model2=copy.deepcopy(model)
config=ActorConfig(padding_free=False,use_torch_compile=False)
actor=DataParallelPPOActor(config,model);actor2=DataParallelPPOActor(config,model2)
ids=torch.randint(101,(3,8))
batch={"input_ids":ids,"responses":ids[:,1:],"attention_mask":torch.ones_like(ids),"position_ids":torch.arange(8).expand(3,-1),"response_mask":mask}
os.environ["WM_SPEED_ONLY"]="0";dense=actor._forward_micro_batch(batch,1.0)
os.environ["WM_SPEED_ONLY"]="1";sparse=actor2._forward_micro_batch(batch,1.0)
assert dense.shape==sparse.shape==mask.shape
assert torch.allclose(dense[mask],sparse[mask],atol=1e-6)
(dense*mask*weights).sum().backward();(sparse*mask*weights).sum().backward()
for p1,p2 in zip(model.parameters(),model2.parameters()):
 assert torch.allclose(p1.grad,p2.grad,atol=1e-6)
print("PASS: truncation, visual features, actual log-prob/actor gradients, shifted metrics.")
