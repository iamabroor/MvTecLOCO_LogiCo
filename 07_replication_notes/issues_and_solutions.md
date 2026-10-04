# Issues and solutions

| Problem | Cause | Fix |
|---|---|---|
| Zero-byte DINOv3 weights file | Placeholder URL used in the download command | Deleted the file and re-downloaded with the real link from Meta's email (327 MB) |
| CUDA out of memory (about 28 MiB short) | Batch 14 on 24 GB | Gradient accumulation patch, `--accum_steps 2`, plus `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Disk full at 300 GB, training crashed, desktop went black | Intermediate checkpoints accumulated | Deleted intermediate checkpoints, retrained the affected category, restarted the workspace |
| SSH "key not found" | Wrong working directory | `cd` to the folder holding the `.pem` |
| SSH "UNPROTECTED PRIVATE KEY FILE" (Windows) | Extra users in the key file's ACL | Reset permissions with `icacls` so only my user has access; the 60-second window had to be reactivated |
| Matplotlib label overlap in figures | Default placement | Moved legend and value labels |
| `putpalette` error in visualiser | Palette applied to wrong image mode | Build a "P" mode image and use `putdata` |
