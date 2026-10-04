# Scripts

| File | Purpose |
|---|---|
| `accum.patch`, `apply_accum_patch.sh` | Gradient accumulation (2 x 7 = batch 14) so training fits in 24 GB |
| `run_all_categories.sh` | Train and evaluate all five categories |
| `make_charts.py` | The three comparison figures in `05_results/figures/` |
| `make_qualitative.py` | Paper-style heatmap grid (image, ground truth, best checkpoint, epoch 200) |
| `visualize_logico.py` | 10 test images with their IDs, scores and per-branch maps; writes a score CSV for every test image |

`visualize_logico.py` examples (from the cloned repo):
```bash
python visualize_logico.py --category breakfast_box                  # 10 random images
python visualize_logico.py --category breakfast_box --select worst   # failure cases
python visualize_logico.py --category breakfast_box --ids logical_anomalies/017 good/003
```
