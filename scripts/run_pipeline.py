"""Run train -> detect -> official eval in one go (verbose).

Defaults assume dev set is under data/raw/development/Development_Set with
Training_Set and Validation_Set subfolders.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import os

def run(cmd: list[str]) -> None:
    print("\n[CMD]", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    parser = argparse.ArgumentParser(description="End-to-end few-shot pipeline runner")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--train_annotations", default="data/raw/development/Development_Set/Training_Set")
    parser.add_argument("--val_annotations", default="data/raw/development/Development_Set/Validation_Set")
    parser.add_argument("--val_audio", default="data/raw/development/Development_Set/Validation_Set")
    parser.add_argument("--checkpoint", default="experiments/proto.ckpt")
    parser.add_argument("--preds", default="preds_val.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--team_name", default="TESTteam")
    parser.add_argument("--savepath", default=".")
    parser.add_argument("--chunk_size", type=int, default=8, help="Chunk size for query windows during training")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    args = parser.parse_args()

    py = sys.executable

    if args.verbose:
        print(f"[pipeline] config={args.config}")
        print(f"[pipeline] train_ann={args.train_annotations}")
        print(f"[pipeline] val_ann={args.val_annotations}")
        print(f"[pipeline] val_audio={args.val_audio}")
        print(f"[pipeline] device={args.device} chunk_size={args.chunk_size} epochs={args.epochs}")
        print(f"[pipeline] checkpoint={args.checkpoint} preds={args.preds}")

    # 1) Train
    train_cmd = [
        py,
        "train_events.py",
        "--config",
        args.config,
        "--annotations_root",
        args.train_annotations,
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--checkpoint_out",
        args.checkpoint,
        "--save_best",
        "--chunk_size",
        str(args.chunk_size),
    ]
    if args.verbose:
        train_cmd.append("--verbose")
    run(train_cmd)

    # 2) Detect on validation
    detect_cmd = [
        py,
        "run_fewshot.py",
        "--config",
        args.config,
        "--annotations_root",
        args.val_annotations,
        "--audio_root",
        args.val_audio,
        "--output_csv",
        args.preds,
        "--device",
        args.device,
        "--checkpoint",
        args.checkpoint,
    ]
    if args.verbose:
        detect_cmd.append("--verbose")
    run(detect_cmd)

    # 3) Official eval
    eval_script = Path("src/eval/dcase/evaluation_metrics/evaluation.py")
    # if the script isn't in CWD (e.g., running from a different working dir), try absolute path from repo root
    candidates = [eval_script, Path(__file__).resolve().parent.parent / "src/eval/dcase/evaluation_metrics/evaluation.py"]
    script_path = None
    for c in candidates:
        if c.exists():
            script_path = c
            break
    if script_path is None:
        print(f"[pipeline] WARNING: evaluation script not found at {candidates}")
    else:
        run(
            [
                py,
                str(script_path),
                "-pred_file",
                args.preds,
                "-ref_files_path",
                args.val_annotations,
                "-team_name",
                args.team_name,
                "-dataset",
                "VAL",
                "-savepath",
                args.savepath,
            ]
        )


if __name__ == "__main__":
    main()
