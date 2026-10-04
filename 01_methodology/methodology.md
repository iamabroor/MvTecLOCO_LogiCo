# LogiCo methodology (summary)

**Problem.** Industrial images can be wrong in two ways. *Structural* anomalies are local defects such as scratches
or dents. *Logical* anomalies break a rule about the scene: a component is missing, extra, in the wrong place or the
wrong count. Logical anomalies look locally normal, so patch-based methods miss them.

**Idea.** Map pre-trained features into a component-level feature space, and reconstruct at both the component
level and the patch level. Train on normal images only.

## Pipeline

1. **Frozen encoder.** DINOv3 ViT-B/16 on 512x512 images; features from layer 11, aggregated over multiple crops.
2. **Component segmentation.** Open-vocabulary segmentation with dino.txt, using a per-category list of text
   prompts (`configs/segmentations/mvtec_loco/<category>.yaml` in the authors' repo). Gives a component map.
3. **Component prototypes.** The mean feature of each component.
4. **Three branches.**
   - **CRN** (component reconstruction network): reconstructs component features, so wrong or missing
     components give a large error. Targets logical anomalies.
   - **SRN** (patch-level reconstruction network): reconstructs patch features. Targets structural anomalies.
   - **SMD** (segmentation-map discriminator): judges the segmentation map itself, which catches count anomalies.
5. **Training on normal images only**, with synthetic component edits (delete, replace, cut-paste, erase) to give
   the networks something to reject.
6. **Fusion.** Each branch map is z-normalised with statistics from the validation set, the maps are max-fused and
   Gaussian-smoothed.
7. **Image score** = mean of the top 0.5% of the fused map.

## Training recipe (official)

200 epochs, AdamW, lr 2e-4 to 1e-4 with warm-up and cosine decay, weight decay 1e-4, batch 14, seed 42,
evaluation every 5 epochs.

## Metric

Mean of logical AUROC and structural AUROC per category, averaged over categories (`compute_imagewise_log_metrics`
in the authors' code).

## Diagram

See `pipeline_diagram.png`.
