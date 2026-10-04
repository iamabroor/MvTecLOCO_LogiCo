# Implementation notes

- Code: official repo, commit `ac724ffb0c36392b443eecc41b58ba152b4298a0`, with one patch (gradient accumulation).
- Dataset: MVTec LOCO, five categories, with the authors' pre-made segmentation maps.
- Hardware: single A10G 24 GB.
- Seed 42, one run per category (no repeated seeds), so no error bars.

## Deviations from the paper

1. Gradient accumulation 2 x 7 instead of a single batch of 14 (memory).
2. Evaluation batch 7 (memory).
3. splicing_connectors was retrained from scratch after a disk-full crash at epoch 35.
4. Checkpoint "best" is chosen on the test set (official protocol). Epoch-200 numbers are provided as the
   unselected alternative.

## Interpretation

Mean 95.66 (best) is 0.64 below the paper's 96.3; mean 93.88 (epoch 200) is 2.42 below. The gap is consistent with
the checkpoint-selection effect, single seed and accumulation. screw_bag is the least stable category.
