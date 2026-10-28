"""CPU tests for mathematical identity, conditioning, missing rows, state restoration and geometry."""
import copy, json
from pathlib import Path
import numpy as np
import torch
from ola import OnlineStats, source_loss, target_loss
torch.set_num_threads(4)
torch.manual_seed(19)
s=OnlineStats("cpu")
b=torch.tensor([-.2,.1,.0709,.4,3.6,.4,2.1215,-.5])
z=torch.randn(1,8,12,12,requires_grad=True)
q=torch.softmax(torch.randn_like(z)+b[None,:,None,None],1)
mask=torch.rand(1,12,12)>.3
old=(q*(q.log()-(z+b[None,:,None,None]).log_softmax(1))).sum(1)[mask].mean()
new,_=target_loss(z,q,mask,b,s,0)
go=torch.autograd.grad(old,z,retain_graph=True)[0]
gn=torch.autograd.grad(new,z)[0]
err=float((go-gn).abs().max())
assert err<1e-7,err
empty,_=target_loss(z,q,torch.zeros_like(mask),b,s,.1)
assert empty.item()==0 and torch.autograd.grad(empty,z)[0].abs().sum()==0
sy=torch.randint(0,8,(1,12,12));sy[:,:2]=255
l,_=source_loss(z,sy,s,.1) # Missing tables MUST produce zero correction.
ref=torch.nn.functional.cross_entropy(z,sy,ignore_index=255)
assert torch.allclose(l,ref,atol=1e-7)
allignored,_=source_loss(z,torch.full_like(sy,255),s,.1)
assert allignored.item()==0
for i in range(8):
    yy=torch.arange(32*32).reshape(1,32,32)%8
    raw=torch.randn(1,8,32,32)+torch.arange(8)[None,:,None,None]*.2
    for domain in ("source","target"):s.collect(domain,raw,yy,yy!=255,i)
s.fit()
assert all(s.summary()["ready_rows"]["source"]) and all(s.summary()["ready_rows"]["target"])
assert all(torch.isfinite(v).all() for t in s.tables.values() for v in t.values())
clone=OnlineStats("cpu");clone.load_state_dict(s.state_dict())
raw=torch.randn(67,8);lab=torch.arange(67)%8
a=s.correction("source",raw,lab,.1);a2=clone.correction("source",raw,lab,.1)
assert torch.equal(a,a2) and a.abs().max()<=.5
assert torch.equal(s.generator.get_state(),clone.generator.get_state())
# Source CE is identical to one-hot target objective when domains and bias match.
clone.tables["target"]=clone.tables["source"]
zz=torch.randn(1,8,7,7,requires_grad=True);yy=torch.arange(49).reshape(1,7,7)%8
onehot=torch.nn.functional.one_hot(yy,8).permute(0,3,1,2).float()
ls,_=source_loss(zz,yy,clone,.1)
lt,_=target_loss(zz,onehot,yy>=0,torch.zeros(8),clone,.1)
assert torch.allclose(ls,lt,atol=2e-7),(ls.item(),lt.item())
# Analytic target gradient: mean of row probabilities minus q; correction is detached.
qq=torch.softmax(torch.randn_like(zz),1)
mask=yy>=0
loss,_=target_loss(zz,qq,mask,b,clone,.1)
grad=torch.autograd.grad(loss,zz)[0].permute(0,2,3,1).reshape(-1,8)
zf=zz.detach().permute(0,2,3,1).reshape(-1,8)
qf=qq.permute(0,2,3,1).reshape(-1,8)
expected=-qf
for c in range(8):
    a=clone.correction("target",zf,torch.full((len(zf),),c,dtype=torch.long),.1)
    expected=expected+qf[:,c,None]*(zf+b-a).softmax(-1)
expected/=len(zf)
assert (expected-grad).abs().max()<1e-7
# All eight D4 transforms invert on asymmetric coordinates.
x=torch.arange(5*7).reshape(1,1,5,7)
for view in range(8):
    k,h=divmod(view,2)
    y=torch.rot90(x,k,(-2,-1))
    if h:y=y.flip(-1)
    if h:y=y.flip(-1)
    y=torch.rot90(y,-k,(-2,-1))
    assert torch.equal(x,y)
result={"status":"PASS","device":"CPU","alpha_zero_gradient_error":err,
        "conditioned_soft_gradient_error":float((expected-grad).abs().max()),
        "checks":["zero-strength KL-gradient identity","empty target mask","ignore255",
          "missing-table zero fallback","GMM finite","statistics roundtrip",
          "onehot limit","conditional soft gradient","D4 all eight inverse"],
        "accuracy_claim":False}
Path(__file__).with_name("preflight_cpu.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))

