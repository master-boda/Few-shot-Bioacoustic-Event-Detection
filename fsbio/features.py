"""feature extraction utilities used by the baseline."""

from __future__ import annotations

import os
from glob import glob
from itertools import chain
from typing import List, Tuple

import h5py
import librosa
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None


def _time_to_frame(df: pd.DataFrame, fps: float) -> Tuple[List[int], List[int]]:
    # add a small margin around each event
    df.loc[:, 'Starttime'] = df['Starttime'] - 0.025
    df.loc[:, 'Endtime'] = df['Endtime'] + 0.025
    start_time = [int(np.floor(start * fps)) for start in df['Starttime']]
    end_time = [int(np.floor(end * fps)) for end in df['Endtime']]
    return start_time, end_time


def _create_patches(
    df_pos: pd.DataFrame,
    pcen: np.ndarray,
    glob_cls_name: str,
    file_name: str,
    hf: h5py.File,
    seg_len: int,
    hop_seg: int,
    fps: float,
) -> List[str]:
    # chunk the time-freq rep into fixed windows
    if len(hf['features'][:]) == 0:
        file_index = 0
    else:
        file_index = len(hf['features'][:])

    start_time, end_time = _time_to_frame(df_pos, fps)

    # for csv files with a column name call, use the global class name
    if 'CALL' in df_pos.columns:
        cls_list = [glob_cls_name] * len(start_time)
    else:
        cls_list = [df_pos.columns[(df_pos == 'POS').loc[index]].values for index, _ in df_pos.iterrows()]
        cls_list = list(chain.from_iterable(cls_list))

    label_list: List[str] = []
    for index in range(len(start_time)):
        str_ind = start_time[index]
        end_ind = end_time[index]
        label = cls_list[index]

        # extract segments with hop_seg stride
        if end_ind - str_ind > seg_len:
            shift = 0
            while end_ind - (str_ind + shift) > seg_len:
                pcen_patch = pcen[int(str_ind + shift):int(str_ind + shift + seg_len)]
                hf['features'].resize((file_index + 1, pcen_patch.shape[0], pcen_patch.shape[1]))
                hf['features'][file_index] = pcen_patch
                label_list.append(label)
                file_index += 1
                shift = shift + hop_seg

            pcen_patch_last = pcen[end_ind - seg_len:end_ind]
            hf['features'].resize((file_index + 1, pcen_patch.shape[0], pcen_patch.shape[1]))
            hf['features'][file_index] = pcen_patch_last
            label_list.append(label)
            file_index += 1
        else:
            # if patch is shorter than seg_len, tile it
            pcen_patch = pcen[str_ind:end_ind]
            if pcen_patch.shape[0] == 0:
                print(pcen_patch.shape[0])
                print("The patch is of 0 length")
                continue

            repeat_num = int(seg_len / (pcen_patch.shape[0])) + 1
            pcen_patch_new = np.tile(pcen_patch, (repeat_num, 1))
            pcen_patch_new = pcen_patch_new[0:int(seg_len)]
            hf['features'].resize((file_index + 1, pcen_patch_new.shape[0], pcen_patch_new.shape[1]))
            hf['features'][file_index] = pcen_patch_new
            label_list.append(label)
            file_index += 1

    print("Total files created : {}".format(file_index))
    return label_list


class FeatureExtractor:
    # pcen mel feature extractor

    def __init__(self, conf):
        self.sr = conf.features.sr
        self.n_fft = conf.features.n_fft
        self.hop = conf.features.hop_mel
        self.n_mels = conf.features.n_mels
        self.fmax = conf.features.fmax

    def extract_feature(self, audio: np.ndarray) -> np.ndarray:
        # librosa>=0.10 enforces keyword-only args
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=self.n_mels,
            fmax=self.fmax,
        )
        pcen = librosa.core.pcen(mel_spec, sr=22050)
        return pcen.astype(np.float32)


def _extract_feature(audio_path: str, extractor: FeatureExtractor, conf) -> np.ndarray:
    y, _ = librosa.load(audio_path, sr=conf.features.sr)
    # scaling audio as per librosa docs
    y = y * (2 ** 32)
    pcen = extractor.extract_feature(y)
    return pcen.T


def feature_transform(conf, mode: str = 'train'):
    # main feature extraction entry
    fps = conf.features.sr / conf.features.hop_mel
    seg_len = int(round(conf.features.seg_len * fps))
    hop_seg = int(round(conf.features.hop_seg * fps))
    extension = "*.csv"

    extractor = FeatureExtractor(conf)

    if mode == 'train':
        print("=== Processing training set ===")
        meta_path = conf.path.train_dir
        all_csv_files = [
            file
            for path_dir, _, _ in os.walk(meta_path)
            for file in glob(os.path.join(path_dir, extension))
        ]
        # use the full training set
        hdf_tr = os.path.join(conf.path.feat_train, 'Mel_train.h5')
        hf = h5py.File(hdf_tr, 'w')
        hf.create_dataset(
            'features',
            shape=(0, seg_len, conf.features.n_mels),
            maxshape=(None, seg_len, conf.features.n_mels),
        )
        label_tr: List[List[str]] = []

        for file in all_csv_files:
            split_list = file.split('/')
            glob_cls_name = split_list[split_list.index('Training_Set') + 1]
            file_name = split_list[split_list.index('Training_Set') + 2]
            df = pd.read_csv(file, header=0, index_col=False)
            audio_path = file.replace('csv', 'wav')
            print("Processing file name {}".format(audio_path))

            pcen = _extract_feature(audio_path, extractor, conf)
            df_pos = df[(df == 'POS').any(axis=1)]
            label_list = _create_patches(df_pos, pcen, glob_cls_name, file_name, hf, seg_len, hop_seg, fps)
            label_tr.append(label_list)

        print(" Feature extraction for training set complete")
        num_extract = len(hf['features'])
        flat_list = [item for sublist in label_tr for item in sublist]
        hf.create_dataset('labels', data=[s.encode() for s in flat_list], dtype='S20')
        data_shape = hf['features'].shape
        hf.close()
        return num_extract, data_shape

    if mode == 'eval':
        print("=== Processing Validation set ===")
        meta_path = conf.path.eval_dir
        all_csv_files = [
            file
            for path_dir, _, _ in os.walk(meta_path)
            for file in glob(os.path.join(path_dir, extension))
        ]
        num_extract = 0

        for file in all_csv_files:
            split_list = file.split('/')
            eval_file_name = split_list[split_list.index('Validation_Set') + 1]
            eval_file_name = eval_file_name + "/" + split_list[split_list.index('Validation_Set') + 2]
            hdf_eval = os.path.join(conf.path.feat_eval, eval_file_name.replace('csv', 'h5'))
            os.makedirs(os.path.dirname(hdf_eval), exist_ok=True)
            hf = h5py.File(hdf_eval, 'w')

            audio_path = file.replace('csv', 'wav')
            pcen = _extract_feature(audio_path, extractor, conf)

            df_eval = pd.read_csv(file, header=0, index_col=False)
            start_time = df_eval['Starttime'].values.astype(float)
            end_time = df_eval['Endtime'].values.astype(float)
            label_list = df_eval['Q'].values

            index_sup = np.where(label_list == 'POS')[0][:conf.train.n_shot]
            if len(index_sup) == 0:
                hf.close()
                continue
            max_len = max(end_time[index_sup] - start_time[index_sup])
            seg_len_eval = int(round(max_len * fps))
            if seg_len_eval <= 0:
                hf.close()
                continue
            hop_seg_eval = int(round(seg_len_eval / 2))

            print("Segment length for file is {}".format(seg_len_eval))
            print("Creating negative dataset")
            print("Creating Positive dataset")
            print("Creating query dataset")

            hf.create_dataset('feat_pos', shape=(0, conf.features.n_mels, seg_len_eval), maxshape=(None, conf.features.n_mels, seg_len_eval))
            hf.create_dataset('feat_neg', shape=(0, conf.features.n_mels, seg_len_eval), maxshape=(None, conf.features.n_mels, seg_len_eval))
            hf.create_dataset('feat_query', shape=(0, conf.features.n_mels, seg_len_eval), maxshape=(None, conf.features.n_mels, seg_len_eval))

            # support features
            for index in range(len(index_sup)):
                start_idx = int(round(start_time[index_sup[index]] * fps))
                end_idx = int(round(end_time[index_sup[index]] * fps))
                if end_idx - start_idx > seg_len_eval:
                    while start_idx + seg_len_eval <= end_idx:
                        spec = pcen[start_idx:start_idx + seg_len_eval].T
                        start_idx += hop_seg_eval
                        hf['feat_pos'].resize((hf['feat_pos'].shape[0] + 1), axis=0)
                        hf['feat_pos'][-1] = spec
                else:
                    spec = pcen[end_idx - seg_len_eval:end_idx].T
                    hf['feat_pos'].resize((hf['feat_pos'].shape[0] + 1), axis=0)
                    hf['feat_pos'][-1] = spec

            # negative features from full audio
            curr_t0 = 0
            last_frame = pcen.shape[-1]
            while curr_t0 + seg_len_eval <= last_frame:
                spec = pcen[curr_t0:curr_t0 + seg_len_eval].T
                hf['feat_neg'].resize((hf['feat_neg'].shape[0] + 1), axis=0)
                hf['feat_neg'][-1] = spec
                curr_t0 = curr_t0 + hop_seg_eval

            # query features after the shots
            strt_index_query = int(round(end_time[index_sup[-1]] * fps))
            hf.create_dataset('start_index_query', data=[strt_index_query])
            curr_frame = strt_index_query

            while curr_frame + seg_len_eval <= last_frame:
                spec = pcen[curr_frame:curr_frame + seg_len_eval].T
                hf['feat_query'].resize((hf['feat_query'].shape[0] + 1), axis=0)
                hf['feat_query'][-1] = spec
                curr_frame = curr_frame + hop_seg_eval

            num_extract = num_extract + len(hf['feat_query'])
            hf.create_dataset('hop_seg', data=[hop_seg_eval])
            hf.close()

        return num_extract

    raise ValueError("mode must be 'train' or 'eval'")
