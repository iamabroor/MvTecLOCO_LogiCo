"""
visualize_steps.py - save EVERY pipeline step of LogiCo as a separate image, for chosen test images.

For each image it writes a folder  figures/steps_<category>_<type>_<NNN>/  containing:
  01_rgb_input.png               the test image
  02_dinov3_features.png         frozen DINOv3 ViT-B/16 patch features (PCA -> RGB)
  03_component_segmentation.png  open-vocabulary component map (dino.txt prompts) + legend
  04_component_prototypes.png    each component painted with its mean (prototype) feature (PCA -> RGB)
  05_CRN_reconstruction.png      what CRN reconstructs from the prototypes (PCA -> RGB)
  06_CRN_logical_map.png         CRN anomaly map (logical)
  07_SRN_reconstruction.png      what SRN reconstructs (PCA -> RGB)
  08_SRN_structural_map.png      SRN anomaly map (structural)
  09_SMD_count_map.png           SMD anomaly map (count)
  10_fused_anomaly_map.png       max-fused, smoothed map
  11_final_result.png            score, threshold, Normal/ANOMALY verdict, ground truth
  00_summary_all_steps.png       all of the above in one strip

usage (from ~/logico_replication/LogiCo, env activated):
  python visualize_steps.py --category breakfast_box                       # 1 good + 1 logical + 1 structural
  python visualize_steps.py --category breakfast_box --ids logical_anomalies/017 good/003
  python visualize_steps.py --category pushpins --ckpt LogiCo_epoch_200_ckpt.pkl
"""
import os, argparse, warnings
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml
from easydict import EasyDict
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

from utils.utils import get_dinov3_weight_name
from dinov3.hub.backbones import load_dinov3_model
from utils.dataset import get_data_transforms, LOCODatasetWithSeg
from models import LogiCo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

p = argparse.ArgumentParser()
p.add_argument("--category", default="breakfast_box")
p.add_argument("--ckpt", default="LogiCo_best_ckpt.pkl")
p.add_argument("--ids", nargs="*", default=None, help="e.g. logical_anomalies/017 good/003")
p.add_argument("--save_dir", default="saved_results")
p.add_argument("--out_dir", default="figures")
args = p.parse_args()
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
cat = args.category

# ---------------- config, data, model (exactly as evaluation.py) ----------------
cfg = EasyDict(yaml.load(open("configs/train/mvtec_loco.yaml"), Loader=yaml.FullLoader))
seg_cfg = EasyDict(yaml.load(open(f"configs/segmentations/mvtec_loco/{cat}.yaml"), Loader=yaml.FullLoader))
comp_names = list(seg_cfg["prompts"]); palette = list(seg_cfg["palette"])
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

def topk_score(m):  # image score exactly as evaluation.py
    m = F.interpolate(m, size=cfg.resize_mask, mode="bilinear", align_corners=False).flatten(1)
    k = max(1, int(m.shape[1] * cfg.max_ratio))
    return torch.sort(m, dim=1, descending=True)[0][:, :k].mean(1)

want = None
if args.ids:
    want = {i if i.endswith(".png") else i + ".png" for i in args.ids}
seen_types = set()

# ---------------- run the model; keep heavy tensors only for the chosen images ----------------
recs, p99 = [], []
with torch.no_grad():
    for data in loader:
        feats = data["feat"].to(dev); seg = data["seg_map"].to(dev).float()
        B, C, H, W = feats.shape
        size = data["image"].shape[-1]
        rs = model.resize_seg_map(seg, size=(H, W))
        comp_feats = model.lookup(rs, model.get_comp_prototypes(feats, rs))      # prototype-painted features
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
        final = model(data)
        score = topk_score(final)
        for i in range(B):
            path = data["image_path"][i]
            rid = f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}"
            typ = data["defect_type"][i]
            r = dict(id=rid, path=path, type=typ, label=int(data["label"][i]), score=float(score[i]))
            p99.append(float(np.percentile(final[i, 0].cpu().numpy(), 99.9)))
            pick = (rid in want) if want is not None else (typ not in seen_types)
            if pick:
                seen_types.add(typ)
                r.update(feat=feats[i].cpu().numpy(), comp=comp_feats[i].cpu().numpy(),
                         crn_f=crn[i].cpu().numpy(), srn_f=srn[i].cpu().numpy(),
                         crn=crn_m[i, 0].cpu().numpy(), smd=smd_m[i, 0].cpu().numpy(),
                         srn=srn_m[i, 0].cpu().numpy(), final=final[i, 0].cpu().numpy(),
                         gt=(data["gt"][i, 0].numpy() > 0.5), seg=data["seg_map"][i].numpy(),
                         crn_s=float(topk_score(crn_m[i:i+1])), smd_s=float(topk_score(smd_m[i:i+1])),
                         srn_s=float(topk_score(srn_m[i:i+1])))
            recs.append(r)

# ---------------- sanity check + threshold ----------------
lab = np.array([r["label"] for r in recs]); sc = np.array([r["score"] for r in recs]); typ = np.array([r["type"] for r in recs])
g = typ == "good"
auc_l = roc_auc_score(lab[g | (typ == "logical_anomalies")], sc[g | (typ == "logical_anomalies")])
auc_s = roc_auc_score(lab[g | (typ == "structural_anomalies")], sc[g | (typ == "structural_anomalies")])
pr, rc, th = precision_recall_curve(lab, sc)
thr = float(th[np.argmax((2 * pr * rc / np.maximum(pr + rc, 1e-12))[:-1])])
print(f"{cat} [{args.ckpt}] logical {100*auc_l:.2f}  structural {100*auc_s:.2f}  mean {50*(auc_l+auc_s):.2f}  (must match the evaluation log)")
print(f"threshold (F1-max): {thr:.3f}")
chosen = [r for r in recs if "feat" in r]
if want:
    miss = want - {r["id"] for r in chosen}
    if miss: print("not found:", sorted(miss))
vmax = float(np.percentile(p99, 95))

# ---------------- helpers ----------------
S = cfg.image_size
def pca_rgb(ref, *others):
    """PCA fitted on the original DINOv3 features; the same basis is applied to every other feature map,
    so colours are comparable across steps."""
    C, H, W = ref.shape
    pca = PCA(n_components=3).fit(ref.reshape(C, -1).T)
    lo = pca.transform(ref.reshape(C, -1).T).min(0); hi = pca.transform(ref.reshape(C, -1).T).max(0)
    out = []
    for f in (ref,) + others:
        t = (pca.transform(f.reshape(C, -1).T) - lo) / np.maximum(hi - lo, 1e-8)
        t = np.clip(t, 0, 1).reshape(H, W, 3)
        out.append(np.array(Image.fromarray((t * 255).astype(np.uint8)).resize((S, S), Image.NEAREST)))
    return out

def seg_rgb(seg):
    pal = (palette + [0] * 768)[:768]
    im = Image.new("P", (seg.shape[1], seg.shape[0])); im.putdata(seg.astype(np.uint8).flatten().tolist()); im.putpalette(pal)
    return np.array(im.convert("RGB").resize((S, S), Image.NEAREST))

def save_tile(path, title, img, hm=None, note=None, legend=None):
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    ax.imshow(img)
    if hm is not None:
        ax.imshow(np.clip(hm, 0, vmax), cmap="jet", vmin=0, vmax=vmax, alpha=0.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=12)
    if note: ax.set_xlabel(note, fontsize=11)
    if legend:
        ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

# ---------------- write one folder per image ----------------
for r in chosen:
    name = f"steps_{cat}_{r['type']}_{os.path.splitext(os.path.basename(r['id']))[0]}"
    d = os.path.join(args.out_dir, name); os.makedirs(d, exist_ok=True)
    img = np.array(Image.open(r["path"]).convert("RGB").resize((S, S)))
    f_feat, f_comp, f_crn, f_srn = pca_rgb(r["feat"], r["comp"], r["crn_f"], r["srn_f"])
    pred = r["score"] >= thr; ok = pred == bool(r["label"])
    verdict = ("ANOMALY" if pred else "NORMAL") + (" (correct)" if ok else " (WRONG)")
    pal3 = (palette + [0] * 768)
    legend = [Patch(color=np.array(pal3[3*k:3*k+3]) / 255, label=n) for k, n in enumerate(comp_names)]
    gt_show = np.stack([r["gt"] * 255] * 3, -1).astype(np.uint8)
    tiles = [
        ("01_rgb_input", "1. RGB input", img, None, r["id"], None),
        ("02_dinov3_features", "2. Frozen DINOv3 features (PCA)", f_feat, None, "ViT-B/16, layer 11, frozen", None),
        ("03_component_segmentation", "3. Component segmentation", seg_rgb(r["seg"]), None, "open-vocabulary, dino.txt prompts", legend),
        ("04_component_prototypes", "4. Component prototypes (PCA)", f_comp, None, "each component = its mean feature", None),
        ("05_CRN_reconstruction", "5. CRN reconstruction (PCA)", f_crn, None, "rebuilt from prototypes only", None),
        ("06_CRN_logical_map", "6. CRN anomaly map (logical)", img, r["crn"], f"branch score {r['crn_s']:.2f}", None),
        ("07_SRN_reconstruction", "7. SRN reconstruction (PCA)", f_srn, None, "patch-level reconstruction", None),
        ("08_SRN_structural_map", "8. SRN anomaly map (structural)", img, r["srn"], f"branch score {r['srn_s']:.2f}", None),
        ("09_SMD_count_map", "9. SMD anomaly map (count)", img, r["smd"], f"branch score {r['smd_s']:.2f}", None),
        ("10_fused_anomaly_map", "10. Fused anomaly map (max)", img, r["final"], f"image score {r['score']:.2f}", None),
        ("11_final_result", "11. Result", img, r["final"], f"score {r['score']:.2f} | thr {thr:.2f} | {verdict}", None),
    ]
    for fn, title, im, hm, note, lg in tiles:
        save_tile(os.path.join(d, fn + ".png"), title, im, hm, note, lg)
    # ground truth next to result
    save_tile(os.path.join(d, "12_ground_truth_mask.png"), "Ground truth (defect mask)", gt_show, None,
              f"true label: {'anomaly' if r['label'] else 'normal'}")
    # summary strip
    allt = tiles + [("gt", "Ground truth", gt_show, None, "", None)]
    cols = 6; rows = int(np.ceil(len(allt) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.ravel(): ax.axis("off")
    for ax, (fn, title, im, hm, note, lg) in zip(axes.ravel(), allt):
        ax.imshow(im)
        if hm is not None: ax.imshow(np.clip(hm, 0, vmax), cmap="jet", vmin=0, vmax=vmax, alpha=0.5)
        ax.set_title(title, fontsize=8.5); ax.text(0.5, -0.04, note, transform=ax.transAxes, ha="center", va="top", fontsize=7)
    fig.suptitle(f"LogiCo pipeline - {cat} - {r['id']} - {verdict} - checkpoint {args.ckpt}", fontsize=12)
    fig.tight_layout(h_pad=3.5, rect=(0, 0, 1, 0.96)); fig.savefig(os.path.join(d, "00_summary_all_steps.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
    print("wrote", d, "->", verdict, f"score {r['score']:.3f}")
