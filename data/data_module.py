import os
import logging
import numpy as np
import torch
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from os import path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from paths import DATASET_PATHS_DICT
from data.loader import LMDB_Folder_Dataset
from data.data_utils import dict2tensor, get_sample_whitening_fn
from data.batching import pad_batch, BatchIdxSampler_Class
from config import CONFIG


logger = logging.getLogger("logger")


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=None):
        super().__init__()
        self.batch_size = CONFIG.TRAIN.BATCH_SIZE if batch_size is None else batch_size

        self.dataset_path = DATASET_PATHS_DICT[CONFIG.DATASET.NAME]
        self.lmdb_path = os.path.join(self.dataset_path, "lmdb_univl")

        # def setup(self, stage=None):
        transformations = []
        if CONFIG.DATASET.WHITENING:
            whitening_params = np.load(
                os.path.join(self.dataset_path, "whitening_params.npy"), allow_pickle=True
            ).item()
            transformations.append(get_sample_whitening_fn(whitening_params))

        transformations.append(dict2tensor)
        transform = Compose(transformations)

        self.train_dataset = LMDB_Folder_Dataset(
            self.lmdb_path, split="train", transform=transform, activity_type="primary"
        )
        self.val_dataset = LMDB_Folder_Dataset(self.lmdb_path, split="val", transform=transform)
        self.test_dataset = LMDB_Folder_Dataset(self.lmdb_path, split="test", transform=transform)

        logger.info(f"Dataset size | Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, collate_fn=pad_batch, batch_size=self.batch_size, num_workers=32)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, collate_fn=pad_batch, batch_size=self.batch_size, num_workers=32)
        return val_loader

    def test_dataloader(self):
        val_loader = DataLoader(self.test_dataset, collate_fn=pad_batch, batch_size=self.batch_size, num_workers=32)
        return val_loader


def rename_dict(d):
    for key in list(d.keys()):
        if "frame" in key:
            new_k = key.replace("frame", "video")
            d[new_k] = d.pop(key)
        if "step" in key:
            new_k = key.replace("step", "text")
            d[new_k] = d.pop(key)
    return d


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="dbg", help="name of the experiment")
    parser.add_argument("--dataset", type=str, default="CrossTask", help="name of the dataset we are encoding")
    parser.add_argument("--config_path", type=str, default="../conf/datasets/", help="path to config file")
    parser.add_argument("--override", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # Enabling reproducibility
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)
    CONFIG.setup(args.name, args.config_path, args.dataset, override_args=args.override)

    data = DataModule()
    # test video loader
    count = 0
    for sample in data.train_dataloader():
        """
        sample contains following items:
            name: name of the example
            cls: class (or activity) of the video
            cls_name: name of the class (or activity)
            num_frames: Number of frames N
            frames: all N frames in the video, has size [N, H,W, 3]
            frame_LABELs: framewsie labels of the video, has size [N,]
        """
        # import ipdb; ipdb.set_trace()
        # print(sample['name'])
        print(sample["cls"])
        print(sample["num_steps"])
        print(sample["num_frames"])
        print(sample["frame_features"].shape)
        print(sample["step_ids"])
        # print(sample['frame_pad_mask'])
        # print(sample['step_pad_mask'])
        # print(sample)
        input("check")
        count += sample["num_frames"].shape[0]
    print("total num vids is {}".format(count))


if __name__ == "__main__":
    main()
