# Builds paper-style (Fig. 4) qualitative figures from LogiCo's saved vis images.
# usage: python make_qualitative.py [examples_per_category] [logical_anomalies|structural_anomalies]
import os, sys, glob
from PIL import Image, ImageDraw, ImageFont
K = int(sys.argv[1]) if len(sys.argv) > 1 else 1
DEFECT = sys.argv[2] if len(sys.argv) > 2 else "logical_anomalies"
ROOT = "saved_results/mvtec_loco"
CATS = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
T, LABW, HEAD, GAP = 256, 230, 60, 6

def font(sz):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(f): return ImageFont.truetype(f, sz)
    try: return ImageFont.load_default(size=sz)
    except TypeError: return ImageFont.load_default()

def split(path):
    im = Image.open(path).convert("RGB"); w = im.width // 3
    return [im.crop((i * w, 0, (i + 1) * w, im.height)).resize((T, T)) for i in range(3)]  # image, heatmap, gt

cols = ["Image", "Ground truth", "LogiCo (best ckpt)", "LogiCo (epoch 200)"]
rows = []
for c in CATS:
    files = sorted(glob.glob(f"{ROOT}/{c}/vis/{DEFECT}/*.png"))
    if not files: print("no vis images for", c); continue
    idx = [round(i * (len(files) - 1) / max(K, 1)) for i in range(K)] if K > 1 else [len(files) // 2]
    for i in idx:
        name = os.path.basename(files[i])
        img, heat_best, gt = split(files[i])
        last = f"{ROOT}/{c}/vis_epoch200/{DEFECT}/{name}"
        heat_last = split(last)[1] if os.path.exists(last) else Image.new("RGB", (T, T), (230, 230, 230))
        rows.append((f"{c}\n{name}", [img, gt, heat_best, heat_last]))

W = LABW + len(cols) * (T + GAP)
H = HEAD + len(rows) * (T + GAP)
fig = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(fig)
for j, t in enumerate(cols):
    d.text((LABW + j * (T + GAP) + T // 2, HEAD // 2), t, fill=(11, 11, 11), font=font(20), anchor="mm")
for r, (lab, tiles) in enumerate(rows):
    y = HEAD + r * (T + GAP)
    d.multiline_text((LABW - 12, y + T // 2), lab, fill=(60, 60, 60), font=font(17), anchor="rm", align="right")
    for j, t in enumerate(tiles): fig.paste(t, (LABW + j * (T + GAP), y))
os.makedirs("figures", exist_ok=True)
out = f"figures/qualitative_{DEFECT}_{K}per_category.png"
fig.save(out); print("saved", out, fig.size)
