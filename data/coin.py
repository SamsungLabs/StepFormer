import os
import logging
import numpy as np
import torch
import random
import pytorch_lightning as pl
import webdataset as wds
import torch.nn.functional as F
from braceexpand import braceexpand
from os import path as osp
import sys
import warnings
import json
from glob import glob
import pandas as pd
from torch.utils.data import Dataset, DataLoader

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from data.crosstask import FolderDataset
from models.model_utils import subsample_video
from paths import DATASET_PATHS_DICT, HOWTO100M, COIN_PATH
from config import CONFIG
from utils import log_info


logger = logging.getLogger("logger")


def process_fn(sample):
    video_features, json = sample["video_features"], sample["json"]
    if video_features.shape[0] > CONFIG.DATASET.MAX_VIDEO_LEN:
        video_features, json = subsample_video(video_features, CONFIG.DATASET.MAX_VIDEO_LEN, "mean", json=json)
    sample["video_features"] = video_features
    sample["json"] = json
    return sample


class COINModule(pl.LightningDataModule):
    def __init__(self, batch_size=None, fType="univl", num_gpus=1):
        super().__init__()
        self.batch_size = 1

        if CONFIG.DATASET.FTYPE == "UniVL":
            self.train_folders = [f for f in glob(os.path.join(COIN_PATH, "samples_univl")) if os.path.isdir(f)]
            self.val_folders = [f for f in glob(os.path.join(COIN_PATH, "samples_univl")) if os.path.isdir(f)]
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

    def filter_filenames(self, dataset, mode):
        with open(os.path.join(COIN_PATH, "train_names.txt"), "r") as f:
            lines = f.read().split("\n")
            if len(lines) > 0 and lines[-1] == "":
                lines = lines[:-1]
        train_set = set(lines)

        old_filelist_size = len(dataset.filenames)
        new_filelist = []
        for f in dataset.filenames:
            name = f.split("/")[-1].split(".")[0]
            if mode == "train":
                if name in train_set:
                    new_filelist.append(f)
            elif mode == "val":
                if name not in train_set:
                    new_filelist.append(f)
        dataset.filenames = new_filelist
        print(f"COIN {mode}: size {len(new_filelist)} (from {old_filelist_size})")

    def train_dataloader(self):
        random.shuffle(self.train_folders)
        self.train_dataset = FolderDataset(self.train_folders, mode="train", process=process_fn)
        self.filter_filenames(self.train_dataset, mode="train")
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=12,
            collate_fn=self.pad_batch,
            shuffle=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        self.val_dataset = FolderDataset(self.val_folders, mode="val", process=process_fn)
        self.filter_filenames(self.val_dataset, mode="val")
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=12, collate_fn=self.pad_batch, drop_last=True
        )
        return val_loader


def main():
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="dbg", help="name of the experiment")
    parser.add_argument("--model_name", type=str, default="TransformerQueryDecoder", help="name of the dataset we are encoding")
    parser.add_argument("--config_path", type=str, default="./conf/models/", help="path to config file")
    parser.add_argument("--override", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # fmt: on

    # Enabling reproducibility
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    CONFIG.setup(args.name, args.config_path, args.model_name, override_args=args.override)

    data = TarDataModule()

    # test video loader
    for sample in data.train_dataloader():
        """
        sample contains following items:
            name: name of the example
            video_features: tensor of video features, has size [N, 768]
            text_features: tensor of text features, has size [K, 768]
            json: with all text info including "text", "start" and "end" times
        """
        print(sample["name"])
        print(sample["video_features"].shape)
        print(sample["video_pad_mask"].shape)
        print(sample["text_features"].shape)
        print(sample["text_pad_mask"].shape)
        input("check")


if __name__ == "__main__":
    main()
