import os
from glob import glob

import h5py
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from fsbio.data import DataBuilder
from fsbio.features import feature_transform
from fsbio.metrics import evaluate_prototypes, prototypical_loss
from fsbio.model import build_encoder
from fsbio.sampler import EpisodicBatchSampler


# simple seeding helper

def init_seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


# training loop for episodic proto

def train_protonet(encoder, train_loader, valid_loader, conf, num_batches_tr, num_batches_vd):
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

    for epoch in range(conf.train.epochs):
        print("Epoch {}".format(epoch))
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

        if avg_acc_vd > best_val_acc:
            print("Saving the best model with valdation accuracy {}".format(avg_acc_vd))
            best_val_acc = avg_acc_vd
            torch.save({'encoder': encoder.state_dict()}, best_model_path)

    torch.save({'encoder': encoder.state_dict()}, last_model_path)
    return best_val_acc, encoder


@hydra.main(config_name="config")
def main(conf: DictConfig):
    # ensure folders exist
    os.makedirs(conf.path.feat_path, exist_ok=True)
    os.makedirs(conf.path.feat_train, exist_ok=True)
    os.makedirs(conf.path.feat_eval, exist_ok=True)

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
        init_seed()

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

        best_acc, _ = train_protonet(encoder, train_loader, valid_loader, conf, num_episodes_tr, num_episodes_vd)
        print("Best accuracy of the model on training set is {}".format(best_acc))

    if conf.set.eval:
        device = conf.train.device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        init_seed()

        name_arr = np.array([])
        onset_arr = np.array([])
        offset_arr = np.array([])
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

            onset, offset = evaluate_prototypes(conf, hdf_eval, device, strt_index_query)
            hdf_eval.close()

            name_arr = np.append(name_arr, np.repeat(audio_name, len(onset)))
            onset_arr = np.append(onset_arr, onset)
            offset_arr = np.append(offset_arr, offset)

        if len(name_arr) > 0:
            out_arr = np.vstack((name_arr, onset_arr, offset_arr)).T
            out_path = os.path.join(conf.path.root_dir, 'eval_output.csv')
            np.savetxt(out_path, out_arr, delimiter=',', fmt='%s', header='Audiofilename,Starttime,Endtime', comments='')
        else:
            print('No detections found')


if __name__ == '__main__':
    main()
