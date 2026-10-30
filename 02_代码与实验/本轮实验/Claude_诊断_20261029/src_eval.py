"""Source1399 D4-TTA8 mIoU for one checkpoint under given bias JSONs (collapse detector only;
Source ranking is known to anti-correlate with Target). Usage: src_eval.py CKPT OUT.json BIAS.json [...]"""
import sys
sys.path.insert(0, "/home/jhy/aic2026_h76_20261026/code/reliable_kd_20261028")
from rkd_common import *  # noqa: E402,F401

ckpt, out, biases = sys.argv[1], sys.argv[2], sys.argv[3:]
B = {Path(b).stem: torch.tensor(json.load(open(b))["bias"], dtype=torch.float32).cuda()[:, None, None] for b in biases}
B["zero"] = torch.zeros(8, 1, 1).cuda()
val = [x.strip() for x in open("/home/jhy/uav_semantic_segmentation/splits/val.txt").read().split() if x.strip()]
model = model_from(ckpt); model.eval()
cm = {k: np.zeros((8, 8), np.int64) for k in B}
for n in val:
    n = n if n.endswith(".png") else n + ".png"
    x = normalize_to_tensor(read_rgb(SOURCE_DATA / "train" / "images" / n)).unsqueeze(0).cuda()
    z = predict_raw_tta(model, x)[0]
    gt = torch.from_numpy(np.asarray(Image.open(SOURCE_DATA / "train" / "masks" / n)).astype(np.int64)).cuda()
    v = gt >= 1
    for k, b in B.items():
        p = F.interpolate((z + b)[None], size=gt.shape, mode="bilinear", align_corners=False)[0].argmax(0)
        cm[k] += torch.bincount((gt[v] - 1) * 8 + p[v], minlength=64).reshape(8, 8).cpu().numpy()
res = {}
for k, c in cm.items():
    iou = np.diag(c) / np.maximum(c.sum(0) + c.sum(1) - np.diag(c), 1)
    res[k] = {"miou": round(100 * iou.mean(), 4), "iou": (100 * iou).round(2).tolist()}
json.dump({"ckpt": ckpt, "n": len(val), **res}, open(out, "w"), indent=1)
print(json.dumps({k: v["miou"] for k, v in res.items()}))
