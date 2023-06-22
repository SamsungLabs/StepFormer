import os
import logging
import numpy as np
import torch
import random
import pytorch_lightning as pl
from glob import glob

import webdataset as wds
import torch.nn.functional as F
from braceexpand import braceexpand
from os import path as osp
import sys

from torch.utils.data import Dataset, DataLoader

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from paths import CT_PATH
from config import CONFIG
from utils import log_info


logger = logging.getLogger("logger")


class FolderDataset(Dataset):
    def __init__(self, folders, mode, process=None):
        self.mode = mode
        self.process = process

        folders = [folders] if type(folders) == str else folders
        self.filenames = []
        for folder in folders:
            self.filenames.extend(glob(os.path.join(folder, "*.npy")))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        sample = self.decode(self.filenames[idx])
        sample = self.to_torch(sample)
        if self.process is not None:
            sample = self.process(sample)
        return sample

    def decode(self, path):
        sample_dict = np.load(path, allow_pickle=True).item()
        return sample_dict

    def to_torch(self, sample):
        for key, val in sample.items():
            sample[key] = torch.tensor(val) if type(val) == np.ndarray else val

        # mapping torch tensors to float32
        for key, val in sample.items():
            if type(val) == torch.Tensor:
                sample[key] = val.to(torch.float32)
        return sample


class CrossTaskModule(pl.LightningDataModule):
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

        if CONFIG.DATASET.FTYPE == "UniVL":
            self.train_folders = [f for f in glob(os.path.join(CT_PATH, "univl_folders/train/*")) if os.path.isdir(f)]
            self.val_folders = [f for f in glob(os.path.join(CT_PATH, "univl_folders/val/*")) if os.path.isdir(f)]
            print(self.val_folders)
        elif CONFIG.DATASET.FTYPE == "MIL-NCE":
            raise NotImplementedError

    def pad_batch(self, batch):
        name = [s["name"] for s in batch]
        features_v = [s["video_features"] for s in batch]
        features_t = [s["text_features"] for s in batch]
        json_t = [s["json"] for s in batch]

        max_len_v = max([v.shape[0] for v in features_v])
        padded_features_v = [F.pad(v, (0, 0, 0, max_len_v - v.shape[0]), value=0) for v in features_v]

        max_len_t = max([t.shape[0] for t in features_t])
        padded_features_t = [F.pad(t, (0, 0, 0, max_len_t - t.shape[0]), value=0) for t in features_t]

        # getting the length in json
        for i in range(len(features_v)):
            json_t[i]["len"] = len(features_v[i].squeeze())

        batch_dict = dict()
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

        batch_dict["video_features"] = torch.stack(padded_features_v)
        batch_dict["video_pad_mask"] = torch.stack(masks_v)

        batch_dict["text_features"] = torch.stack(padded_features_t)
        batch_dict["text_pad_mask"] = torch.stack(masks_t)

        batch_dict["name"] = name
        batch_dict["json"] = json_t
        return batch_dict

    def train_dataloader(self, num_workers=32):
        random.shuffle(self.train_folders)
        self.train_dataset = FolderDataset(self.train_folders, mode="train")
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            collate_fn=self.pad_batch,
            shuffle=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self, num_workers=12):
        self.val_dataset = FolderDataset(self.val_folders, mode="train")
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            collate_fn=self.pad_batch,
            drop_last=True,
        )
        return val_loader


class CrossTask_old(pl.LightningDataModule):
    def __init__(self, batch_size=1, fType="univl"):
        super().__init__()
        self.batch_size = batch_size

        train_tar_paths = os.path.join(CT_PATH, "new_shards/full_univl-{000000..000025}.tar")
        if fType == "univl":
            val_tar_paths = os.path.join(CT_PATH, "new_shards/full_univl-000000.tar")
        elif fType == "mil-nce":
            val_tar_paths = os.path.join(CT_PATH, "mil_nce_shards/val/full_mil_nce_512.tar")
        self.train_urls = list(braceexpand(train_tar_paths))
        self.val_urls = list(braceexpand(val_tar_paths))

    def get_tar_dataset(self, shards, mode="train"):
        self.mode = mode
        process_np = lambda t: torch.from_numpy(t).to(torch.float32)

        wds_dataset = (
            wds.WebDataset(shards, shardshuffle=False)
            .decode()
            .rename(name="__key__", video_features="video.npy", text_features="text.npy", json="json")
            .to_tuple("name", "video_features", "text_features", "json")
            .map_tuple(None, process_np, process_np, None)
            .batched(self.batch_size, collation_fn=self.pad_batch, partial=True)
        )
        return wds_dataset

    def pad_batch(self, batch):
        # name, features_v, features_t, json_t = list(zip(*list(batch_values)))
        name, features_v, features_t, json_t = list(zip(*list(batch)))
        max_len_v = max([v.shape[0] for v in features_v])
        padded_features_v = [F.pad(v, (0, 0, 0, max_len_v - v.shape[0]), value=0) for v in features_v]

        max_len_t = max([t.shape[0] for t in features_t])
        padded_features_t = [F.pad(t, (0, 0, 0, max_len_t - t.shape[0]), value=0) for t in features_t]

        batch_dict = dict()
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

        batch_dict["video_features"] = torch.stack(padded_features_v)
        batch_dict["video_pad_mask"] = torch.stack(masks_v)

        batch_dict["text_features"] = torch.stack(padded_features_t)
        batch_dict["text_pad_mask"] = torch.stack(masks_t)

        batch_dict["name"] = name
        batch_dict["json"] = json_t

        return batch_dict

    def train_dataloader(self):
        self.train_dataset = self.get_tar_dataset(self.train_urls, mode="train")
        train_loader = wds.WebLoader(self.train_dataset, num_workers=0, collate_fn=None)
        return train_loader

    def val_dataloader(self):
        self.val_dataset = self.get_tar_dataset(self.val_urls, mode="val")
        val_loader = wds.WebLoader(self.val_dataset, num_workers=0, collate_fn=None)
        return val_loader
