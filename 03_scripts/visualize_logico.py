"""
visualize_logico.py - qualitative figures for the LogiCo replication (MVTec LOCO).

For N test images it shows, per row:
  Image | Components (segmentation map) | Logical branch (CRN) | Count branch (SMD) |
  Structural branch (SRN) | Final anomaly map | Ground truth
and labels each row with the image's ID in the test set (e.g. logical_anomalies/017.png),
its image score, the decision threshold and whether the prediction is correct.

It also writes a CSV with the score of EVERY test image, so any image number can be looked up.

usage (from ~/logico_replication/LogiCo, env activated):
  python visualize_logico.py --category breakfast_box                     # 10 random images (2 good, 4 logical, 4 structural)
  python visualize_logico.py --category breakfast_box --select best       # most confident correct detections
  python visualize_logico.py --category breakfast_box --select worst      # failure cases
  python visualize_logico.py --category breakfast_box --ids logical_anomalies/017 good/003
  python visualize_logico.py --category breakfast_box --ckpt LogiCo_epoch_200_ckpt.pkl
"""
import os, argparse, warnings, random, csv
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml
from easydict import EasyDict
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve
warnings.filterwarnings("ignore")

from utils.utils import get_dinov3_weight_name
from dinov3.hub.backbones import load_dinov3_model
from utils.dataset import get_data_transforms, LOCODatasetWithSeg
from models import LogiCo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--category", default="breakfast_box")
p.add_argument("--ckpt", default="LogiCo_best_ckpt.pkl")
p.add_argument("--select", default="random", choices=["random", "best", "worst"])
p.add_argument("--n", type=int, default=10)
p.add_argument("--ids", nargs="*", default=None, help="e.g. logical_anomalies/017 good/003")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--save_dir", default="saved_results")
p.add_argument("--out_dir", default="figures")
args = p.parse_args()
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
cat = args.category

# ---------------- config, data, model (exactly as evaluation.py) ----------------
cfg = EasyDict(yaml.load(open("configs/train/mvtec_loco.yaml"), Loader=yaml.FullLoader))
seg_cfg = EasyDict(yaml.load(open(f"configs/segmentations/mvtec_loco/{cat}.yaml"), Loader=yaml.FullLoader))
comp_names = seg_cfg["prompts"]
palette = list(seg_cfg["palette"])
tf, gt_tf = get_data_transforms(cfg.image_size)
test = LOCODatasetWithSeg(root=f"data/mvtec_loco/{cat}", image_size=cfg.image_size, transform=tf,
                          gt_transform=gt_tf, phase="test")
enc = load_dinov3_model(cfg.encoder, pretrained_weight_path=os.path.join("ckpt", get_dinov3_weight_name(cfg.encoder)))
enc.name = cfg.encoder; enc.to(dev).eval()
test.extract_feats(enc, cfg.encoder_target_layers, cfg.encoder_feat_stride, cfg.encoder_feat_dim,
                   device=dev, use_aggregate_features=cfg.use_aggregate_features, use_cache=True)
loader = torch.utils.data.DataLoader(test, batch_size=7, shuffle=False, num_workers=4)
model = LogiCo(**cfg.model, encoder=enc, comp_num=len(comp_names), encoder_feat_dim=cfg.encoder_feat_dim, device=dev)
model.load_ckpt(os.path.join(args.save_dir, "mvtec_loco", cat, args.ckpt))
model.eval()

def topk_score(m):  # image score exactly as evaluation.py: mean of top max_ratio of the map at resize_mask
    m = F.interpolate(m, size=cfg.resize_mask, mode="bilinear", align_corners=False).flatten(1)
    k = max(1, int(m.shape[1] * cfg.max_ratio))
    return torch.sort(m, dim=1, descending=True)[0][:, :k].mean(1)

# ---------------- run every test image, keep branch maps ----------------
recs = []
with torch.no_grad():
    for data in loader:
        feats = data["feat"].to(dev); seg = data["seg_map"].to(dev).float()
        B, C, H, W = feats.shape
        size = data["image"].shape[-1]
        rs = model.resize_seg_map(seg, size=(H, W))
        comp_feats = model.lookup(rs, model.get_comp_prototypes(feats, rs))
        crn, hs = model.CRN(comp_feats)
        srn = model.SRN(feats, hs[::-1])
        z = lambda m, mu, sd: (F.interpolate(m, size=(size, size), mode="bilinear", align_corners=True) - mu) / sd
        crn_m = z(model.cal_anomaly_maps(feats, crn), model.CRN_mean, model.CRN_std)
        srn_m = z(model.cal_anomaly_maps(feats, srn), model.SRN_mean, model.SRN_std)
        if model.comp_num != 1:
            oh = model.get_one_hot_encoding(model.resize_seg_map(seg, size=model.SMD_size))
            smd_m = z(nn.Sigmoid()(model.SMD(oh)), model.SMD_mean, model.SMD_std)
        else:
            smd_m = torch.zeros_like(crn_m)
        final = model(data)                       # the model's own fused + smoothed map
        score = topk_score(final)
        for i in range(B):
            path = data["image_path"][i]
            recs.append(dict(id=f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
                             path=path, type=data["defect_type"][i], label=int(data["label"][i]),
                             score=float(score[i]), crn_s=float(topk_score(crn_m[i:i+1])),
                             smd_s=float(topk_score(smd_m[i:i+1])), srn_s=float(topk_score(srn_m[i:i+1])),
                             crn=crn_m[i, 0].cpu().numpy(), smd=smd_m[i, 0].cpu().numpy(),
                             srn=srn_m[i, 0].cpu().numpy(), final=final[i, 0].cpu().numpy(),
                             gt=(data["gt"][i, 0].numpy() > 0.5), seg=data["seg_map"][i].numpy()))

# ---------------- sanity check + decision threshold ----------------
lab = np.array([r["label"] for r in recs]); sc = np.array([r["score"] for r in recs])
typ = np.array([r["type"] for r in recs])
g = typ == "good"
auc_l = roc_auc_score(lab[g | (typ == "logical_anomalies")], sc[g | (typ == "logical_anomalies")])
auc_s = roc_auc_score(lab[g | (typ == "structural_anomalies")], sc[g | (typ == "structural_anomalies")])
pr, rc, th = precision_recall_curve(lab, sc)
f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
thr = float(th[np.argmax(f1[:-1])])
print(f"{cat} [{args.ckpt}]  logical AUROC {100*auc_l:.2f}  structural {100*auc_s:.2f}  "
      f"mean {50*(auc_l+auc_s):.2f}   (should match the evaluation log)")
print(f"decision threshold (F1-max over all {len(recs)} test images): {thr:.3f}")

os.makedirs(args.out_dir, exist_ok=True)
tag = "best" if "best" in args.ckpt else os.path.splitext(args.ckpt)[0].replace("LogiCo_", "")
csv_path = f"{args.out_dir}/scores_{cat}_{tag}.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["image_id", "type", "label", "score", "flagged_anomaly", "correct",
                                   "logical_branch", "count_branch", "structural_branch"])
    for r in sorted(recs, key=lambda r: r["id"]):
        pred = int(r["score"] >= thr)
        w.writerow([r["id"], r["type"], r["label"], f"{r['score']:.4f}", pred, int(pred == r["label"]),
                    f"{r['crn_s']:.4f}", f"{r['smd_s']:.4f}", f"{r['srn_s']:.4f}"])
print("all test-image scores ->", csv_path)

# ---------------- choose images ----------------
by = {t: [r for r in recs if r["type"] == t] for t in ["good", "logical_anomalies", "structural_anomalies"]}
if args.ids:
    want = [i if i.endswith(".png") else i + ".png" for i in args.ids]
    chosen = [r for w_ in want for r in recs if r["id"] == w_]
    missing = set(want) - {r["id"] for r in chosen}
    if missing: print("not found:", sorted(missing))
else:
    n_good = max(1, round(args.n * 0.2)); n_log = (args.n - n_good + 1) // 2; n_str = args.n - n_good - n_log
    quota = {"good": n_good, "logical_anomalies": n_log, "structural_anomalies": n_str}
    rng = random.Random(args.seed); chosen = []
    for t, k in quota.items():
        pool = by[t]
        if args.select == "random":
            chosen += rng.sample(pool, min(k, len(pool)))
        else:  # best = most confident correct; worst = most wrong
            anomalous = t != "good"
            key = (lambda r: -r["score"]) if (anomalous == (args.select == "best")) else (lambda r: r["score"])
            chosen += sorted(pool, key=key)[:k]

# ---------------- draw ----------------
def seg_rgb(seg):
    pal = (palette + [0] * 768)[:768]
    im = Image.new("P", (seg.shape[1], seg.shape[0])); im.putdata(seg.astype(np.uint8).flatten().tolist()); im.putpalette(pal)
    return np.array(im.convert("RGB"))

vmax = float(np.percentile(np.stack([r["final"] for r in recs]), 99.9))
cols = ["Image", "Components", "Logical (CRN)", "Count (SMD)", "Structural (SRN)", "Final map", "Ground truth"]
fig, axes = plt.subplots(len(chosen), len(cols), figsize=(2.3 * len(cols), 2.55 * len(chosen) + 0.6), squeeze=False)
for r_i, r in enumerate(chosen):
    img = np.array(Image.open(r["path"]).convert("RGB").resize((cfg.image_size, cfg.image_size)))
    tiles = [img, seg_rgb(r["seg"]), r["crn"], r["smd"], r["srn"], r["final"], r["gt"]]
    bscores = [None, None, r["crn_s"], r["smd_s"], r["srn_s"], r["score"], None]
    for c_i, (ax, t) in enumerate(zip(axes[r_i], tiles)):
        if c_i in (2, 3, 4, 5):
            ax.imshow(img); ax.imshow(np.clip(t, 0, vmax), cmap="jet", vmin=0, vmax=vmax, alpha=0.5)
            ax.text(4, cfg.image_size - 8, f"{bscores[c_i]:.2f}", color="white", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.6, pad=1.5, lw=0))
        elif c_i == 6:
            ax.imshow(t, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(t)
        ax.set_xticks([]); ax.set_yticks([])
        if r_i == 0: ax.set_title(cols[c_i], fontsize=10)
    pred = r["score"] >= thr
    ok = pred == bool(r["label"])
    verdict = ("ANOMALY" if pred else "normal") + (" - correct" if ok else " - WRONG")
    axes[r_i][0].set_ylabel(f"#{r_i+1}  {r['id']}\nscore {r['score']:.2f}  (thr {thr:.2f})\n{verdict}",
                            fontsize=8.5, rotation=0, ha="right", va="center", labelpad=8,
                            color="black" if ok else "#c0392b")
sel = "ids" if args.ids else args.select
fig.suptitle(f"LogiCo replication - {cat} - {sel} - checkpoint: {tag}   "
             f"(mean I-AUROC {50*(auc_l+auc_s):.2f}; numbers on maps = branch / image scores)", fontsize=11, y=1.0)
fig.tight_layout()
out = f"{args.out_dir}/vis_{cat}_{sel}_{tag}.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
print("figure ->", out)
