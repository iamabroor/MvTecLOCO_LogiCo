# Training settings used

Official config `configs/train/mvtec_loco.yaml`, unchanged, except where listed under Deviations.

| Setting | Value |
|---|---|
| Encoder | DINOv3 ViT-B/16, frozen, layer 11, aggregated multi-crop features |
| Image size | 512 |
| Epochs | 200 |
| Optimiser | AdamW, weight decay 1e-4 |
| Learning rate | 2e-4 to 1e-4, warm-up then cosine |
| Batch size | 14 effective |
| Seed | 42 |
| Evaluation | every 5 epochs |
| Gradient clipping | 0.1 |
| Image score | mean of top 0.5% of the anomaly map |

## Deviations

- `--accum_steps 2`: micro-batch 7, two steps per update, to avoid CUDA out-of-memory on 24 GB.
  Mathematically close to batch 14 but not bit-identical.
- `--batch_size 7` for evaluation (same memory reason).
