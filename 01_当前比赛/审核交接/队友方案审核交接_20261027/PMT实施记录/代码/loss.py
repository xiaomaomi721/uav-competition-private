"""Valid-pixel Hungarian mask classification; source GT 0=ignore, 1..8=classes."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def sample_points(valid,n):
 return valid[torch.randint(len(valid),(n,),device=valid.device)]
def gather_logits(masks,ids,h,w):
 xy=torch.stack(((ids%w+.5)/w,(ids//w+.5)/h),-1)*2-1
 # [Q,1,Hm,Wm], one shared list or per-query lists
 if xy.ndim==2:xy=xy[None].expand(len(masks),-1,-1)
 return F.grid_sample(masks[:,None].float(),xy[:,None],mode='bilinear',align_corners=False)[:,0,0]
def mask_loss(masks,classes,gt,points=12544):
 assert gt.ndim==2
 valid=(gt.flatten()>0).nonzero().flatten()
 if len(valid)==0:return masks.sum()*0+classes.sum()*0
 labels=torch.unique(gt);labels=labels[labels>0]-1
 assert labels.min()>=0 and labels.max()<8
 h,w=gt.shape;ids=sample_points(valid,points)
 with torch.no_grad():
  pred=gather_logits(masks,ids,h,w)
  target=(gt.flatten()[ids][None]==(labels[:,None]+1)).float()
  ce=F.softplus(pred).mean(1)[:,None]-pred@target.T/points
  prob=pred.sigmoid()
  dice=1-(2*prob@target.T+1)/(prob.sum(1)[:,None]+target.sum(1)[None]+1)
  cost=2*(-classes.float().softmax(-1)[:,labels])+5*ce+5*dice
  assert torch.isfinite(cost).all()
  qi,ti=linear_sum_assignment(cost.cpu().numpy())
  qi=torch.as_tensor(qi,device=masks.device);ti=torch.as_tensor(ti,device=masks.device)
 matched=masks[qi]
 with torch.no_grad():
  candidate=sample_points(valid,points*3)
  uncertain=-gather_logits(matched,candidate,h,w).abs()
  hard=candidate[uncertain.topk(int(points*.75),dim=1).indices]
  rand=sample_points(valid,len(qi)*(points-hard.shape[1])).reshape(len(qi),-1)
  chosen=torch.cat((hard,rand),1)
  truth=(gt.flatten()[chosen]==(labels[ti,None]+1)).float()
 pred=gather_logits(matched,chosen,h,w)
 bce=F.binary_cross_entropy_with_logits(pred,truth,reduction='none').mean(1).mean()
 prob=pred.sigmoid()
 dice=(1-(2*(prob*truth).sum(1)+1)/(prob.sum(1)+truth.sum(1)+1)).mean()
 class_target=torch.full((len(classes),),8,dtype=torch.long,device=masks.device)
 class_target[qi]=labels[ti]
 weights=torch.ones(9,device=masks.device);weights[-1]=.1
 ce=F.cross_entropy(classes.float(),class_target,weight=weights)
 return 2*ce+5*bce+5*dice

def test():
 torch.manual_seed(42)
 gt=torch.zeros(32,32,dtype=torch.long);gt[8:24,8:24]=1;gt[16:24,8:24]=2
 m=torch.randn(5,32,32,requires_grad=True);c=torch.randn(5,9,requires_grad=True)
 state=torch.get_rng_state();a=mask_loss(m,c,gt,128)
 altered=m.detach().clone();altered[:,gt==0]=1000
 torch.set_rng_state(state);b=mask_loss(altered,c,gt,128)
 assert torch.allclose(a,b,atol=1e-6),(a,b)
 a.backward();assert m.grad[:,gt==0].abs().max()==0
 assert m.grad.abs().sum()>0 and torch.isfinite(m.grad).all()
 zero=mask_loss(m,c,torch.zeros_like(gt),128);assert zero==0
 print('PASS: ignored pixels excluded from matching/loss; finite backward; all-ignore safe',flush=True)
if __name__=='__main__':test()

