import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# validated reference palette (light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF})

cats = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
best = {"L": [99.02, 99.82, 99.39, 88.21, 92.29], "S": [98.70, 97.19, 91.58, 91.51, 98.86], "M": [98.86, 98.51, 95.49, 89.86, 95.58]}
last = {"L": [99.02, 99.51, 98.88, 78.25, 90.94], "S": [98.70, 95.87, 87.82, 92.34, 97.46], "M": [98.86, 97.69, 93.35, 85.29, 94.20]}
salad = {"L": [89.72, 99.64, 99.92, 82.27, 94.90], "S": [82.46, 98.96, 97.85, 94.81, 97.45], "M": [86.09, 99.30, 98.89, 88.54, 96.18]}
paper = {"L": 96.5, "S": 96.0, "M": 96.3}
avg = lambda d, k: float(np.mean(d[k]))

def legend_row(ax, items, y=0.985):
    hs = [ax.scatter([], [], color=col, marker=mk, s=70, edgecolors=SURF, linewidths=1.5) for lab, col, mk in items]
    ax.figure.legend(hs, [i[0] for i in items], loc="upper left", bbox_to_anchor=(0.01, y), ncol=len(items), frameon=False, handletextpad=0.3, columnspacing=1.4, fontsize=10.5)

# ---------- Fig A: overall averages, paper vs replication ----------
fig, ax = plt.subplots(figsize=(9, 4.4)); fig.subplots_adjust(top=0.84)
groups = [("Logical", "L"), ("Structural", "S"), ("Mean (reported metric)", "M")]
y = np.arange(len(groups))[::-1]
for yi, (name, k) in zip(y, groups):
    vals = [(paper[k], INK, "D", "Paper"), (avg(best, k), BLUE, "o", "Replication, best ckpt"), (avg(last, k), ORANGE, "s", "Replication, epoch 200")]
    lo, hi = min(v[0] for v in vals), max(v[0] for v in vals)
    ax.plot([lo, hi], [yi, yi], color=GRID, lw=2, zorder=1)
    for v, c, m, _ in vals:
        ax.scatter(v, yi, color=c, marker=m, s=90, zorder=3, edgecolors=SURF, linewidths=2)
    ax.text(avg(best, k), yi + 0.22, f"{avg(best,k):.2f}", ha="center", color=INK2, fontsize=10)
    ax.text(avg(last, k), yi + 0.22, f"{avg(last,k):.2f}", ha="center", color=INK2, fontsize=10)
    ax.text(paper[k], yi + 0.22, f"{paper[k]:.1f}", ha="center", color=INK, fontsize=10, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([g[0] for g in groups])
ax.set_ylim(-0.5, 2.6); ax.set_xlim(92.5, 97.2); ax.set_xlabel("Image AUROC (%), MVTec LOCO, average of 5 categories")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
legend_row(ax, [("Paper (ECCV 2026)", INK, "D"), ("Replication, best checkpoint", BLUE, "o"), ("Replication, epoch 200", ORANGE, "s")])
fig.suptitle("LogiCo replication vs paper: 95.66 vs 96.3 mean AUROC (−0.64)", x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.03)
fig.savefig("fig1_overall_vs_paper.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- Fig B: per category, LogiCo vs SALAD replication ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 5.0), sharey=True); fig.subplots_adjust(top=0.80)
y = np.arange(len(cats))[::-1]
for ax, (title, k) in zip(axes, [("Logical", "L"), ("Structural", "S"), ("Mean", "M")]):
    for yi, b, l, s in zip(y, best[k], last[k], salad[k]):
        ax.plot([min(b, l, s), max(b, l, s)], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.scatter(salad[k], y, color=AQUA, marker="^", s=80, zorder=3, edgecolors=SURF, linewidths=1.5)
    ax.scatter(last[k], y, color=ORANGE, marker="s", s=70, zorder=3, edgecolors=SURF, linewidths=1.5)
    ax.scatter(best[k], y, color=BLUE, marker="o", s=80, zorder=4, edgecolors=SURF, linewidths=1.5)
    for yi, b in zip(y, best[k]):
        ax.text(b, yi + 0.25, f"{b:.1f}", ha="center", fontsize=9, color=INK2)
    ax.axvline(paper[k], color=INK, lw=1, ls=(0, (3, 3)))
    ax.text(paper[k], len(cats) - 0.45, f"paper avg {paper[k]}", ha="center", fontsize=9, color=INK)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xlim(76, 101.5); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    ax.set_ylim(-0.6, len(cats) - 0.1)
axes[0].set_yticks(y); axes[0].set_yticklabels(cats)
axes[1].set_xlabel("Image AUROC (%)")
legend_row(axes[0], y=0.955, items=[("LogiCo, best ckpt", BLUE, "o"), ("LogiCo, epoch 200", ORANGE, "s"), ("SALAD (your replication)", AQUA, "^")])
fig.suptitle("Per category: LogiCo wins big on breakfast_box; screw_bag is hardest for both methods", x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.0)
fig.savefig("fig2_per_category_logico_vs_salad.png", dpi=200, bbox_inches="tight"); plt.close(fig)

# ---------- Fig C: effect of test-set checkpoint selection ----------
gap = np.array(best["M"]) - np.array(last["M"])
fig, ax = plt.subplots(figsize=(9, 4)); 
order = np.argsort(gap)
ycat = np.arange(len(cats))
ax.barh(ycat, gap[order], color=BLUE, height=0.55)
for yi, g in zip(ycat, gap[order]):
    ax.text(g + 0.08, yi, f"+{g:.2f}", va="center", fontsize=10, color=INK2)
ax.set_yticks(ycat); ax.set_yticklabels([cats[i] for i in order])
ax.set_xlabel("Mean AUROC gain from picking the 'best' epoch on the test set (points)")
ax.set_xlim(0, 5.5); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
fig.suptitle(f"Test-set checkpoint selection adds {gap.mean():.2f} points on average, most on screw_bag", x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.02)
fig.savefig("fig3_checkpoint_selection_effect.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("avg", {k: round(avg(best,k),2) for k in "LSM"}, {k: round(avg(last,k),2) for k in "LSM"}, {k: round(avg(salad,k),2) for k in "LSM"}, gap.round(2))
