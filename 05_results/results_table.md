# Results: MVTec LOCO image-level AUROC (%)

Paper averages: logical 96.5, structural 96.0, mean 96.3. The per-category paper numbers were not available to me,
so only averages are compared.

| Category | Logical (best) | Structural (best) | Mean (best) | Mean (epoch 200) |
|---|---|---|---|---|
| breakfast_box | 99.02 | 98.70 | 98.86 | 98.86 |
| juice_bottle | 99.82 | 97.19 | 98.51 | 97.69 |
| pushpins | 99.39 | 91.58 | 95.49 | 93.35 |
| screw_bag | 88.21 | 91.51 | 89.86 | 85.29 |
| splicing_connectors | 92.29 | 98.86 | 95.58 | 94.20 |
| **Average** | **95.75** | **95.57** | **95.66** | **93.88** |

"Best" = checkpoint with the highest test AUROC during training (the official protocol, which selects on the test
set). "Epoch 200" = final checkpoint, no selection. Full numbers, including the SALAD comparison, are in
`LogiCo_paper_vs_replication.csv`.

Gap to paper (mean): -0.64 best checkpoint, -2.42 epoch 200. Weakest category: screw_bag.
