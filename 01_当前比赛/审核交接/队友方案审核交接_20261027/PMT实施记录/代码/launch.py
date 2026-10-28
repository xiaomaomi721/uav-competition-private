"""Stage runner: checksum -> feature equivalence -> real-data smoke -> formal training."""
import os,sys,subprocess,json,time,fcntl,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-bc304d79-c807-968a-eaad-d2c2ecc5a984',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1')
def status(stage,**kw):
 d={'stage':stage,'unix_time':time.time(),**kw};(R/'status.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)
def main():
 lock=open(R/'launch.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,args in [('conversion',['backbone.py',str(R/'dino.safetensors')]),('smoke',['train.py','--mode','smoke']),('formal',['train.py','--mode','train'])]:
  status(name,status='running')
  with open(R/(name+'.log'),'w') as log:
   ret=subprocess.run([sys.executable,*args],cwd=R,env=ENV,stdout=log,stderr=subprocess.STDOUT)
  if ret.returncode:
   status(name,status='failed',exit_code=ret.returncode);sys.exit(ret.returncode)
  status(name,status='completed')
 status('all',status='completed')
if __name__=='__main__':main()

