# Few-shot-Bioacoustic-Event-Detection
Lightweight scaffold for the DCASE 2023 Few-Shot Bioacoustic Event Detection task.

## Layout
- `configs/`: YAML configs (see `configs/baseline.yaml`).
- `src/data/`: dataset and episode samplers.
- `src/models/`: model backbones (proto-net stub).
- `src/training/`: training/validation loop scaffolding.
- `src/utils/`: config utilities.
- `train.py`: entrypoint that wires config, data, and model together.
- `data/`: place raw and processed audio locally (ignored by git).
- `experiments/`: checkpoints and logs (ignored by git).
- `notebooks/`: scratch space for exploration.

## Quick start (synthetic)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py --config configs/baseline.yaml
```

## Real data wiring
1) Download audio for the DCASE 2023 few-shot task (FSDKaggle2018/ESC-50 subsets) into `data/raw/`.
2) Create metadata CSVs with columns `filepath,label,split` (split optional) relative to `data_root`. Place them at `data/metadata/train.csv` and `data/metadata/val.csv` (paths configurable in `configs/baseline.yaml`).
3) Run `train.py` with the baseline config to sample n-way/k-shot episodes and train the prototypical network backbone.
