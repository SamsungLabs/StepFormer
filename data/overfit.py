import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from os import path as osp
import torch.nn.functional as F
from config import CONFIG


class OverfitDataset(Dataset):
    def __init__(self, batch, len):
        self.len = len
        self.batch = batch

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.batch


class OverfitDataModule(pl.LightningDataModule):
    def __init__(self, data_module, len=100):
        super().__init__()
        for train_batch in data_module.train_dataloader():
            break

        for val_batch in data_module.val_dataloader():
            break

        self.train_dataset = OverfitDataset(train_batch, len)
        self.val_dataset = OverfitDataset(val_batch, 1)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, collate_fn=lambda x: x[0], num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, collate_fn=lambda x: x[0], num_workers=1)
