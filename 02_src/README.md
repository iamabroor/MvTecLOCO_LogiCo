# Source code

The method code is the authors' and is not copied here. Get the exact version I used:

```bash
git clone https://github.com/cnulab/LogiCo.git
cd LogiCo
git checkout ac724ffb0c36392b443eecc41b58ba152b4298a0
git apply ../03_scripts/accum.patch      # adds --accum_steps (gradient accumulation)
```

Licence of the authors' code: Apache-2.0. Please cite the paper (arXiv 2606.28688) if you use it.

Not included, download separately:
- MVTec LOCO dataset (from MVTec)
- DINOv3 ViT-B/16 weights (from Meta, requires accepting their licence)
- Segmentation maps (links in the authors' README)
