# Environment (AWS RMIT RACE Workbench)

- Instance: NVIDIA A10G 24 GB, about 1.21 USD per hour. **Stop the workspace when not training.**
- Access: DCV browser desktop, or SSH with the workspace key (activation window is about 60 seconds).
- Disk: 300 GB. Delete intermediate checkpoints (the run script does this) or the disk fills and the desktop breaks.

## Setup

```bash
python -m venv logico-env && source logico-env/bin/activate     # or conda
pip install -r requirements.txt
```
Then follow `02_src/README.md`, download the dataset, segmentation maps and DINOv3 weights per the authors' README,
and run `03_scripts/run_all_categories.sh`.

## Record your exact environment (paste results into this folder)

```bash
nvidia-smi > gpu_info.txt
python --version >> gpu_info.txt
pip freeze > env_freeze.txt
```
