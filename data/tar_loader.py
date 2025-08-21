import os
import logging
import numpy as np
import torch
import random
import pytorch_lightning as pl
import webdataset as wds
import torch.nn.functional as F
from os import path as osp
import sys
import warnings
from glob import glob
from io import BytesIO

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from paths import DATASET_PATHS_DICT, HOWTO100M
from config import CONFIG
from models.model_utils import subsample_video
from utils import log_info, get_world_info


logger = logging.getLogger("logger")


class TarDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=None, dbg=False):
        super().__init__()
        self.dbg = dbg
        self.batch_size = CONFIG.TRAIN.BATCH_SIZE if batch_size is None else batch_size
        self.dataset_path = DATASET_PATHS_DICT[CONFIG.DATASET.NAME]

        if CONFIG.DATASET.FTYPE == "MIL-NCE":
            raise NotImplementedError
        elif CONFIG.DATASET.FTYPE == "UniVL":
            all_folders = glob(os.path.join(HOWTO100M, "univl_tars/*"))
            all_folders = all_folders
            # n_train = len(all_folders)  # FIXME
            n_train = max(len(all_folders) - 8, 2)
            self.train_folders = all_folders[:n_train]
            self.val_folders = all_folders[n_train:]
        else:
            assert f"No such feature type {CONFIG.DATASET.FTYPE}"

        self.max_step_chars = 100
        self.max_step_words = 15

    def get_tar_dataset(self, shards, mode="train"):
        shuffle = 10000 if mode == "train" else 0

        wds_dataset = (
            wds.WebDataset(shards, shardshuffle=shuffle, resampled=shuffle, nodesplitter=wds.shardlists.split_by_node)
            .map(self.read_feature_to_torch)
            .map(self.filter_phrases_and_resample_videos)
            .batched(self.batch_size, collation_fn=self.pad_batch, partial=False)
        )
        return wds_dataset

    def read_feature_to_torch(self, stream):
        features = np.lib.format.read_array(BytesIO(stream["npy"]), allow_pickle=True).item()
        for k, val in features.items():
            if k == "name":
                video_name = val
            elif k == "json":
                annot = val
            elif k == "text_features":
                text_feature = val
            elif k == "video_features":
                video_feature = torch.tensor(val)

        return video_name, annot, text_feature.to(torch.float32), video_feature.to(torch.float32)

    def filter_phrases_and_resample_videos(self, sample):
        # the function filters out phrases that are too long, and resamples long videos
        name, annot, text_features, video_features = sample

        for traverse_direction in ["forward", "reverse"]:
            good_idxs = []
            for i in range(len(annot["phrases"])):
                phrase, start, end = annot["phrases"][i], annot["start"][i], annot["end"][i]
                prev_good_phrase = "@" if len(good_idxs) == 0 else annot["phrases"][good_idxs[-1]]
                pprev_good_phrase = "@" if len(good_idxs) < 2 else annot["phrases"][good_idxs[-2]]
                is_right_len = (
                    len(phrase) <= self.max_step_chars
                    and len(phrase.split(" ")) <= self.max_step_words
                    and len(phrase) > 0
                )
                has_start_end_time = start is not None and end is not None
                is_duplicate = phrase.startswith(prev_good_phrase) or phrase.startswith(pprev_good_phrase)
                if is_right_len and has_start_end_time and not is_duplicate:
                    good_idxs.append(i)

            # writing good_idxs in reversed order, so that to parse it in reverse next time
            annot = {key: [val[i] for i in good_idxs[::-1]] for key, val in annot.items()}
            text_features = text_features[good_idxs[::-1]]

        # trimming videos
        n_clips = video_features.shape[0]
        if n_clips > CONFIG.DATASET.MAX_VIDEO_LEN:
            video_features = subsample_video(video_features, CONFIG.DATASET.MAX_VIDEO_LEN, "mean")
        return text_features, video_features

    def pad_batch(self, batch):
        features_t, features_v = list(zip(*batch))

        # removing first dimension added by the batcher
        if len(features_t[0].shape) == 3:
            features_t = [f[0] for f in features_t]
            features_v = [f[0] for f in features_v]

        max_len_v = max([v.shape[0] for v in features_v])
        padded_features_v = [F.pad(v, (0, 0, 0, max_len_v - v.shape[0]), value=0) for v in features_v]

        max_len_t = max([t.shape[0] for t in features_t])
        padded_features_t = [F.pad(t, (0, 0, 0, max_len_t - t.shape[0]), value=0) for t in features_t]

        # batch_dict = dict()
        masks_v = []
        for f in features_v:
            mask_v = torch.zeros((max_len_v), dtype=bool)
            mask_v[f.shape[0] :] = 1
            masks_v.append(mask_v)

        masks_t = []
        for f in features_t:
            mask_t = torch.zeros((max_len_t), dtype=bool)
            mask_t[f.shape[0] :] = 1
            masks_t.append(mask_t)

        batch = [
            padded_features_v,
            masks_v,
            padded_features_t,
            masks_t,
        ]
        batch = [torch.stack(batch[i]) for i in range(len(batch))]
        return batch

    def train_dataloader(self):
        world_size, rank = get_world_info()
        random.shuffle(self.train_folders)
        shards = self.train_folders[: int(len(self.train_folders) // world_size) * world_size]

        num_batches = len(shards) * 1000 // (self.batch_size * world_size)
        num_batches = 10 if self.dbg else num_batches

        self.train_dataset = self.get_tar_dataset(shards, mode="train")
        train_loader = (
            wds.WebLoader(self.train_dataset, num_workers=12, pin_memory=False, batch_size=None, shuffle=False)
            .repeat(2)
            .with_epoch(num_batches)
            .with_length(num_batches)
        )
        return train_loader

    def val_dataloader(self):
        self.val_dataset = self.get_tar_dataset(self.val_folders, mode="val")
        num_batches = 5 if self.dbg else 50
        val_loader = (
            wds.WebLoader(self.val_dataset, num_workers=12, pin_memory=False, batch_size=None, shuffle=False)
            .with_epoch(num_batches)
            .with_length(num_batches)
        )
        return val_loader
