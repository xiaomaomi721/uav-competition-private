import sys, random
sys.path.insert(0,"/home/jhy/aic2026_claude_diag_20261029")
import mix_train as mt
from rkd_common import *
from src.label_mapping import official_to_train
rows=target_rows(); src_ids=[x.strip() for x in (SRC/"splits"/"train.txt").read_text().split() if x.strip()]
mean,std=mt.MEAN,mt.STD
tiles=[]
for k,(s,t) in enumerate([(src_ids[11],rows[2600]),(src_ids[400],rows[1800]),(src_ids[2000],rows[3100])]):
    rgb=read_rgb(mt.SOURCE_IMG/f"{s}.png"); lab=official_to_train(np.asarray(Image.open(mt.SOURCE_MSK/f"{s}.png")).copy())
    sx=normalize_to_tensor(rgb)[None]; sy=torch.from_numpy(lab.astype(np.int64))
    tx=normalize_to_tensor(read_rgb(TARGET_IMAGES/t["input_name"]))[None]
    M=mt.classmix_mask(sy,random.Random(k))[None,None]
    s01,t01=(sx*std+mean).clamp(0,1),(tx*std+mean).clamp(0,1)
    st=mt.fda(s01,t01,0.01); mix=torch.where(M,st,t01)
    for im in (s01,st,t01,mix):
        tiles.append(Image.fromarray((im[0].permute(1,2,0).numpy()*255).astype(np.uint8)).resize((256,256)))
W=Image.new("RGB",(1024,768))
for i,tl in enumerate(tiles): W.paste(tl,((i%4)*256,(i//4)*256))
W.save("/home/jhy/aic2026_claude_diag_20261029/mixviz.jpg",quality=88); print("ok")
