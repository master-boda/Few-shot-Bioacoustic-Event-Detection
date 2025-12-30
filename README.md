# Few-shot-Bioacoustic-Event-Detection
Lightweight scaffold for the DCASE 2023 Few-Shot Bioacoustic Event Detection task.

## Layout
- `configs/`: YAML configs (see `configs/baseline.yaml`).
- `src/data/`: dataset and episode samplers.
- `src/data/events.py`: per-file 5-shot event dataset (Task 5 style).
- `src/models/`: model backbones (proto-net stub).
- `src/training/`: training/validation loop scaffolding.
- `src/utils/`: config utilities.
- `detect.py`: Task-5-style few-shot event detection script.
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

## Real data wiring (classification scaffold)
1) Download audio for the DCASE 2023 few-shot task (FSDKaggle2018/ESC-50 subsets) into `data/raw/`.
2) Create metadata CSVs with columns `filepath,label,split` (split optional) relative to `data_root`. Place them at `data/metadata/train.csv` and `data/metadata/val.csv` (paths configurable in `configs/baseline.yaml`).
3) Run `train.py` with the baseline config to sample n-way/k-shot episodes and train the prototypical network backbone.

## Task 5-style few-shot event detection
- Annotations: per-file CSVs with `Starttime,Endtime,<label>` where `<label>` is `POS/UNK` (validation/eval format). Place them under a root (e.g., `data/raw/development/Development_Set/Validation_Set`).
- Detection: uses first 5 POS events per file as support, slides windows over the rest, and outputs merged detections.
```
python detect.py --config configs/baseline.yaml \
  --annotations_root data/raw/development/Development_Set/Validation_Set \
  --output_csv predictions.csv \
  --threshold 0.0 \
  --device cuda --batch_size 8   # adjust for your GPU memory
```
Tune `threshold` and `window_seconds/hop_seconds` in the config for your resources.

## Features
- Log-mel spectrograms with optional spectral contrast (controlled via `features.use_spectral_contrast` and `features.spectral_contrast_bands` in `configs/baseline.yaml`). Multi-channel features are stacked along the channel dimension for the Conv2d proto-net.

## Training (per-file 5-shot style)
- `train_events.py` builds support from the first 5 POS events per file and labels query windows after the 5th POS as positive/negative via overlap. This gives a lightweight way to tune the encoder/prototype similarity toward detection.
```
python train_events.py --config configs/baseline.yaml \
  --annotations_root data/raw/development/Development_Set/Training_Set \
  --epochs 5 --device cuda \
  --checkpoint_out experiments/proto.ckpt --save_best
```
Load a checkpoint with `--checkpoint_in`.

## Evaluation
- `evaluate.py` computes an event-based F (Hungarian IoU matching). By default it uses the 5th POS end time from the reference to ignore earlier events.
```
python evaluate.py --pred_csv predictions.csv --ref_csv ground_truth.csv --iou 0.5
```
- `scripts/sweep_threshold.py` can sweep thresholds if you save multiple prediction CSVs (using `{thr}` placeholder in the filename).
- `scripts/run_pipeline.py` runs train -> detect -> official eval in one go (verbose).
- Official DCASE evaluator is vendored at `src/eval/dcase/evaluation_metrics/evaluation.py`. Run from repo root:
```
python src/eval/dcase/evaluation_metrics/evaluation.py \
  -pred_file=preds.csv \
  -ref_files_path=path/to/Validation_Set \
  -team_name=TESTteam \
  -dataset=VAL \
  -savepath=./
```

## Notebooks
- `notebooks/explore_dataset.ipynb` provides a simple dataset inspection (event counts per file and example annotation rows).

## End-to-end few-shot pipeline (Task 5 rules)
- `run_fewshot.py` processes each file independently: first 5 POS per class = support; queries are sliding windows after the 5th POS end; prototypes are class means; scores are cosine distances; post-processing applies threshold/median/min-duration from the config.
- Annotations must have `Starttime,Endtime,label` columns (UNK rows are skipped). Audio basenames must match the annotation basenames.
```
python run_fewshot.py \
  --config configs/baseline.yaml \
  --annotations_root data/raw/development/Development_Set/Validation_Set \
  --audio_root data/raw/development/Development_Set/Validation_Set \
  --output_csv preds.csv \
  --device cuda \
  --checkpoint experiments/proto.ckpt  # optional
```
- Use the official evaluator you copied at `src/eval/dcase/evaluation_metrics/evaluation.py` for scoring.
