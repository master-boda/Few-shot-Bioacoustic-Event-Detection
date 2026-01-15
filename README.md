# Few-shot Bioacoustic Event Detection (ProtoNet + ResNet)

This repo contains a cleaned, reproducible pipeline for the DCASE Few-shot Bioacoustic Event Detection task. The system uses a prototypical network with a ResNet encoder, PCEN-based features, and episodic training. Evaluation supports adaptive segment length, negative sampling from gaps between the 5 shots, optional transductive refinement, and threshold sweeps.

## Pipeline Overview

1) **Features**: Mel spectrogram (128 bins) + PCEN, optionally concatenated with delta-MFCCs.  
2) **Training**: episodic ProtoNet training using a ResNet encoder.  
3) **Evaluation**: adaptive segment length per file, negative prototypes from gaps between the first 5 shots, and optional transductive refinement.  
4) **Post-processing**: optional short-event filtering and gap merging.  

## Quickstart

### Feature extraction
```
python main.py set.features=true set.train=false set.eval=false
```

### Training
```
python main.py set.features=false set.train=true set.eval=false
```

### Evaluation (single threshold)
```
python main.py set.features=false set.train=false set.eval=true
```

### Evaluation with sweep
```
python main.py set.features=false set.train=false set.eval=true \
  eval.sweep=true eval.sweep_start=0.4 eval.sweep_stop=0.8 eval.sweep_step=0.05
```

### Compute official metric
```
python evaluation_metrics/evaluation.py \
  -pred_file=outputs/<run_folder>/eval_output.csv \
  -ref_files_path=/path/to/Development_Set/Validation_Set/ \
  -team_name=TESTteam \
  -dataset=VAL \
  -savepath=outputs/<run_folder>/
```

## Outputs and Logging

Each run creates a folder under `outputs/` with a descriptive name:
```
outputs/train_eval_sweep_td_s2_t1.0_w0.1_YYYYmmdd_HHMMSS/
```

Inside each run folder you will see:
- `eval_output.csv` or `eval_output_thresh_*.csv`
- `train_metrics.csv` and `train_curves.png` (if training)
- `sweep_scores.json` / `sweep_scores.csv` (if sweep)
- `config_snapshot.yaml` (resolved config for the run)

Hydra logs are stored under:
```
outputs/hydra/YYYY-mm-dd/HH-MM-SS/
```

## Scripts

- `scripts/eval_sweep.py`: evaluate all sweep CSVs and generate a PR curve.
- `scripts/tune_transductive.py`: grid search for transductive settings and save a report.
- `scripts/tune_postprocess.py`: grid search for post-processing thresholds and save a report.
- `scripts/tune_eval_stability.py`: grid search for eval samples/iterations with fixed threshold.
- `scripts/tune_hoplen.py`: grid search for eval hop length factor with fixed threshold.
- `scripts/tune_seglen_lim.py`: grid search for eval segment-length limit with fixed threshold.
- `scripts/tune_train.py`: grid search for training tweaks (lr/epochs) and evaluate each model.

Example sweep evaluation:
```
python scripts/eval_sweep.py \
  --sweep_dir=outputs/<run_folder> \
  --ref_files_path=/path/to/Development_Set/Validation_Set/ \
  --team_name=TESTteam \
  --dataset=VAL \
  --plot_pr
```

Example transductive grid search:
```
python scripts/tune_transductive.py \
  --ref_files_path=/path/to/Development_Set/Validation_Set/ \
  --outputs_dir=outputs \
  --steps=1,2,3 \
  --temps=1.0,2.0,3.0 \
  --weights=0.05,0.1,0.2 \
  --sweep_start=0.4 --sweep_stop=0.8 --sweep_step=0.05
```

Example post-processing grid search:
```bash
python scripts/tune_postprocess.py \
  --ref_files_path=/path/to/Development_Set/Validation_Set/ \
  --outputs_dir=outputs \
  --min_fracs=0.05,0.1,0.15,0.2 \
  --merge_fracs=0.05,0.1,0.15,0.2 \
  --sweep_start=0.4 --sweep_stop=0.8 --sweep_step=0.05
```

Example eval stability grid search (fixed threshold):
```bash
python scripts/tune_eval_stability.py \
  --ref_files_path=/path/to/Development_Set/Validation_Set/ \
  --outputs_dir=outputs \
  --samples_neg=30,50,100 \
  --iterations=3,6,10 \
  --threshold=0.6
```

Example hop length factor grid search:
```bash
python scripts/tune_hoplen.py \
  --ref_files_path=/path/to/Development_Set/Validation_Set/ \
  --outputs_dir=outputs \
  --hop_factors=2,3,4 \
  --samples_neg=30 \
  --iterations=6 \
  --threshold=0.6
```

Example segment-length limit grid search:
```bash
python scripts/tune_seglen_lim.py \
  --ref_files_path=/path/to/Development_Set/Validation_Set/ \
  --outputs_dir=outputs \
  --seglen_lims=20,30,40 \
  --hop_factor=2 \
  --samples_neg=30 \
  --iterations=6 \
  --threshold=0.6
```

Example training grid search (lr/epochs):
```bash
python scripts/tune_train.py \
  --ref_files_path=/path/to/Development_Set/Validation_Set/ \
  --outputs_dir=outputs \
  --lrs=0.001,0.0005,0.0001 \
  --epochs=5,10 \
  --threshold=0.6
```

## Configuration Notes

Key config fields in `config.yaml`:

- `features.use_delta_mfcc`, `features.n_mfcc`: add delta-MFCC features.
- `eval.transductive*`: enable and tune transductive refinement.
- `eval.sweep*`: enable threshold sweep and set the range.
- `eval.postprocess`, `eval.min_duration_frac`, `eval.merge_gap_frac`: post-processing.
- `eval.test_seglen_len_lim`, `eval.test_hoplen_fenmu`: adaptive eval segment length.
- `path.output_dir`: base folder for run outputs.
- `path.model_tag`: optional subfolder under `path.Model` for versioned checkpoints.
- `train.seed`, `eval.seed`: random seeds for reproducibility.

Only the **ResNet encoder** is supported in this cleaned version.

## Repo Layout

```
fsbio/                feature extraction, model, sampler, metrics
main.py               entrypoint (features/train/eval)
config.yaml           configuration
evaluation_metrics/   official DCASE evaluation
scripts/              sweep + transductive tuning helpers
outputs/              run outputs (created during execution)
```

## Requirements

Install dependencies with:
```
pip install -r requirements.txt
```

## Post-processing (optional)

Fixed post-processing based on removing events shorter than 200 ms:
```
python post_proc_new.py \
  -val_path=/path/to/Development_Set/Validation_Set/ \
  -evaluation_file=outputs/<run_folder>/eval_output.csv \
  -new_evaluation_file=outputs/<run_folder>/eval_output_post.csv
```

Note: If `eval.postprocess=true` is enabled in config, avoid running `post_proc_new.py` to prevent double filtering.
