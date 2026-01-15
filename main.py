import csv
import json
import os
import re
import random
import sys
from datetime import datetime
from glob import glob

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from fsbio.data import DataBuilder
from fsbio.features import feature_transform
from fsbio.metrics import evaluate_prototypes, prototypical_loss
from fsbio.model import build_encoder
from fsbio.sampler import EpisodicBatchSampler


# simple seeding helper

def init_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# training loop for episodic proto

def _write_train_metrics(run_dir, rows, plot: bool = True):
    if run_dir is None or len(rows) == 0:
        return
    metrics_path = os.path.join(run_dir, "train_metrics.csv")
    with open(metrics_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)
    if not plot:
        return
    try:
        import matplotlib.pyplot as plt

        epochs = [row["epoch"] for row in rows]
        train_loss = [row["train_loss"] for row in rows]
        val_loss = [row["val_loss"] for row in rows]
        train_acc = [row["train_acc"] for row in rows]
        val_acc = [row["val_acc"] for row in rows]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(epochs, train_loss, label="train")
        axes[0].plot(epochs, val_loss, label="val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("epoch")
        axes[0].legend()

        axes[1].plot(epochs, train_acc, label="train")
        axes[1].plot(epochs, val_acc, label="val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "train_curves.png"))
        plt.close(fig)
    except Exception as exc:
        print(f"Failed to plot metrics: {exc}")


def _sanitize_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tag).strip("-")


def _get_run_dir(conf):
    base_dir = conf.path.get("output_dir", os.path.join(conf.path.root_dir, "outputs"))
    os.makedirs(base_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = []
    if conf.set.get("features"):
        parts.append("features")
    if conf.set.get("train"):
        parts.append("train")
    if conf.set.get("eval"):
        parts.append("eval")
    if not parts:
        parts.append("run")
    label = "_".join(parts)
    if conf.eval.get("sweep", False):
        label = f"{label}_sweep"
    run_tag = conf.get("run_tag") or conf.eval.get("run_tag") or conf.train.get("run_tag")
    if run_tag:
        label = f"{label}_{_sanitize_tag(str(run_tag))}"
    run_dir = os.path.join(base_dir, f"{label}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _resolve_model_paths(conf):
    model_tag = conf.path.get("model_tag")
    if not model_tag:
        return
    model_tag = str(model_tag)
    base_model_dir = conf.path.Model
    model_dir = os.path.join(base_model_dir, model_tag)
    conf.path.Model = model_dir
    conf.path.best_model = os.path.join(model_dir, "best_model.pth")
    conf.path.last_model = os.path.join(model_dir, "last_model.pth")


def _write_config_snapshot(conf, run_dir: str):
    if run_dir is None:
        return
    snapshot_path = os.path.join(run_dir, "config_snapshot.yaml")
    with open(snapshot_path, "w") as handle:
        handle.write(OmegaConf.to_yaml(conf, resolve=True))


def _maybe_evaluate_predictions(conf, pred_file, run_dir):
    if not conf.eval.get("compute_metrics", False):
        return
    if conf.eval.get("sweep", False):
        print("Skipping metric evaluation for sweep outputs.")
        return
    eval_dir = os.path.join(conf.path.root_dir, "evaluation_metrics")
    sys.path.insert(0, eval_dir)
    try:
        import evaluation as eval_mod
    finally:
        sys.path.pop(0)
    team = conf.eval.get("metrics_team", "TESTteam")
    dataset = conf.eval.get("metrics_dataset", "VAL")
    eval_mod.evaluate(pred_file, conf.path.eval_dir, team, dataset, run_dir)
    report_glob = os.path.join(run_dir, f"Evaluation_report_{team}_{dataset}_*.json")
    reports = sorted(glob(report_glob))
    if reports:
        with open(reports[-1], "r") as handle:
            report = json.load(handle)
        overall = report.get("overall_scores", {})
        precision = overall.get("precision")
        recall = overall.get("recall")
        fmeasure = overall.get("fmeasure (percentage)")
        if precision is not None and recall is not None and fmeasure is not None:
            print(f"Eval summary: precision={precision} recall={recall} fmeasure%={fmeasure}")


def train_protonet(encoder, train_loader, valid_loader, conf, num_batches_tr, num_batches_vd, run_dir=None):
    if conf.train.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    optim = torch.optim.Adam(encoder.parameters(), lr=conf.train.lr_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optim,
        gamma=conf.train.scheduler_gamma,
        step_size=conf.train.scheduler_step_size,
    )

    best_model_path = conf.path.best_model
    last_model_path = conf.path.last_model
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    best_val_acc = 0.0
    encoder.to(device)
    epoch_rows = []

    for epoch in range(conf.train.epochs):
        print("Epoch {}".format(epoch))
        cur_lr = optim.param_groups[0]["lr"]
        train_iterator = iter(train_loader)
        for batch in tqdm(train_iterator):
            optim.zero_grad()
            encoder.train()

            x, y = batch
            x = x.to(device)
            y = y.to(device)
            x_out = encoder(x)
            tr_loss, tr_acc = prototypical_loss(x_out, y, conf.train.n_shot)
            train_loss.append(tr_loss.item())
            train_acc.append(tr_acc.item())

            tr_loss.backward()
            optim.step()

        avg_loss_tr = np.mean(train_loss[-num_batches_tr:])
        avg_acc_tr = np.mean(train_acc[-num_batches_tr:])
        print('Average train loss: {}  Average training accuracy: {}'.format(avg_loss_tr, avg_acc_tr))
        lr_scheduler.step()
        encoder.eval()

        val_iterator = iter(valid_loader)
        for batch in tqdm(val_iterator):
            x, y = batch
            x = x.to(device)
            x_val = encoder(x)
            valid_loss, valid_acc = prototypical_loss(x_val, y, conf.train.n_shot)
            val_loss.append(valid_loss.item())
            val_acc.append(valid_acc.item())

        avg_loss_vd = np.mean(val_loss[-num_batches_vd:])
        avg_acc_vd = np.mean(val_acc[-num_batches_vd:])
        print('Epoch {}, Validation loss {:.4f}, Validation accuracy {:.4f}'.format(epoch, avg_loss_vd, avg_acc_vd))
        epoch_rows.append(
            {
                "epoch": epoch,
                "lr": cur_lr,
                "train_loss": float(avg_loss_tr),
                "train_acc": float(avg_acc_tr),
                "val_loss": float(avg_loss_vd),
                "val_acc": float(avg_acc_vd),
            }
        )

        if avg_acc_vd > best_val_acc:
            print("Saving the best model with valdation accuracy {}".format(avg_acc_vd))
            best_val_acc = avg_acc_vd
            torch.save({'encoder': encoder.state_dict()}, best_model_path)

    torch.save({'encoder': encoder.state_dict()}, last_model_path)
    if conf.train.get("log_metrics", True):
        _write_train_metrics(run_dir, epoch_rows, plot=bool(conf.train.get("plot_metrics", True)))
    return best_val_acc, encoder


@hydra.main(config_name="config")
def main(conf: DictConfig):
    # ensure folders exist
    os.makedirs(conf.path.feat_path, exist_ok=True)
    os.makedirs(conf.path.feat_train, exist_ok=True)
    os.makedirs(conf.path.feat_eval, exist_ok=True)

    run_dir = None
    if conf.set.train or conf.set.eval:
        run_dir = _get_run_dir(conf)
    _resolve_model_paths(conf)
    _write_config_snapshot(conf, run_dir)

    if conf.set.features:
        print(" --Feature Extraction Stage--")
        num_extract_train, data_shape = feature_transform(conf=conf, mode="train")
        print("Shape of dataset is {}".format(data_shape))
        print("Total training samples is {}".format(num_extract_train))

        num_extract_eval = feature_transform(conf=conf, mode='eval')
        print("Total number of samples used for evaluation: {}".format(num_extract_eval))
        print(" --Feature Extraction Complete--")

    if conf.set.train:
        os.makedirs(conf.path.Model, exist_ok=True)
        hdf_path = os.path.join(conf.path.feat_train, 'Mel_train.h5')
        if not os.path.exists(hdf_path):
            raise FileNotFoundError(
                f"training features not found at {hdf_path}. Run feature extraction or set set.features=true."
            )
        seed = int(conf.train.get("seed", 0))
        init_seed(seed)

        gen_train = DataBuilder(conf)
        x_train, y_train, x_val, y_val = gen_train.generate_train()
        x_tr = torch.tensor(x_train)
        y_tr = torch.LongTensor(y_train)
        x_val = torch.tensor(x_val)
        y_val = torch.LongTensor(y_val)

        samples_per_cls = conf.train.n_shot * 2
        batch_size_tr = samples_per_cls * conf.train.k_way
        batch_size_vd = batch_size_tr

        if conf.train.num_episodes is not None:
            num_episodes_tr = conf.train.num_episodes
            num_episodes_vd = conf.train.num_episodes
        else:
            num_episodes_tr = len(y_train) // batch_size_tr
            num_episodes_vd = len(y_val) // batch_size_vd

        samplr_train = EpisodicBatchSampler(y_train, num_episodes_tr, conf.train.k_way, samples_per_cls)
        samplr_valid = EpisodicBatchSampler(y_val, num_episodes_vd, conf.train.k_way, samples_per_cls)

        train_dataset = torch.utils.data.TensorDataset(x_tr, y_tr)
        valid_dataset = torch.utils.data.TensorDataset(x_val, y_val)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_sampler=samplr_train,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_sampler=samplr_valid,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )

        encoder = build_encoder(conf)

        best_acc, _ = train_protonet(
            encoder,
            train_loader,
            valid_loader,
            conf,
            num_episodes_tr,
            num_episodes_vd,
            run_dir=run_dir,
        )
        print("Best accuracy of the model on training set is {}".format(best_acc))

    if conf.set.eval:
        device = conf.train.device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        seed = int(conf.eval.get("seed", conf.train.get("seed", 0)))
        init_seed(seed)

        name_arr = np.array([])
        onset_arr = np.array([])
        offset_arr = np.array([])
        sweep_enabled = bool(conf.eval.get("sweep", False))
        sweep_data = {}
        all_feat_files = glob(os.path.join(conf.path.feat_eval, '**', '*.h5'), recursive=True)
        if len(all_feat_files) == 0:
            print(f"No evaluation features found under {conf.path.feat_eval}. Run feature extraction first.")
            return

        for feat_file in all_feat_files:
            feat_name = feat_file.split('/')[-1]
            audio_name = feat_name.replace('h5', 'wav')
            print("Processing audio file : {}".format(audio_name))

            hdf_eval = h5py.File(feat_file, 'r')
            strt_index_query = hdf_eval['start_index_query'][:][0]

            onset_offset = evaluate_prototypes(conf, hdf_eval, device, strt_index_query)
            hdf_eval.close()

            if sweep_enabled:
                for thresh, (onset, offset) in onset_offset.items():
                    if thresh not in sweep_data:
                        sweep_data[thresh] = {
                            "name": np.array([]),
                            "onset": np.array([]),
                            "offset": np.array([]),
                        }
                    sweep_data[thresh]["name"] = np.append(
                        sweep_data[thresh]["name"], np.repeat(audio_name, len(onset))
                    )
                    sweep_data[thresh]["onset"] = np.append(sweep_data[thresh]["onset"], onset)
                    sweep_data[thresh]["offset"] = np.append(sweep_data[thresh]["offset"], offset)
            else:
                onset, offset = onset_offset
                name_arr = np.append(name_arr, np.repeat(audio_name, len(onset)))
                onset_arr = np.append(onset_arr, onset)
                offset_arr = np.append(offset_arr, offset)

        if sweep_enabled:
            if len(sweep_data) == 0:
                print('No detections found')
                return
            for thresh, payload in sweep_data.items():
                if len(payload["name"]) == 0:
                    continue
                out_arr = np.vstack((payload["name"], payload["onset"], payload["offset"])).T
                thresh_tag = f"{thresh:.2f}".rstrip("0").rstrip(".").replace(".", "p")
                out_path = os.path.join(run_dir or conf.path.root_dir, f'eval_output_thresh_{thresh_tag}.csv')
                np.savetxt(out_path, out_arr, delimiter=',', fmt='%s', header='Audiofilename,Starttime,Endtime', comments='')
        else:
            if len(name_arr) > 0:
                out_arr = np.vstack((name_arr, onset_arr, offset_arr)).T
                out_path = os.path.join(run_dir or conf.path.root_dir, 'eval_output.csv')
                np.savetxt(out_path, out_arr, delimiter=',', fmt='%s', header='Audiofilename,Starttime,Endtime', comments='')
                _maybe_evaluate_predictions(conf, out_path, run_dir or conf.path.root_dir)
            else:
                print('No detections found')


if __name__ == '__main__':
    main()
