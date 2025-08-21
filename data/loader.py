import sys
import os
import torch
import random
import numpy as np
import lmdb
import pyarrow as pa
import pickle

from os import path as osp
from tqdm import tqdm
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from data.data_utils import Time2FrameNumber
from config import CONFIG


class LMDB_Folder_Dataset(Dataset):
    def __init__(self, folder, split="train", transform=None, activity_type="all", truncate=0):
        self.transform = transform

        # filtering out folders that have desired split
        cls_folders = []
        for cls_folder in glob(osp.join(folder, "*/")):
            files = glob(osp.join(cls_folder, "*.lmdb"))
            file_has_split = ["_{}".format(split) in f for f in files]
            if any(file_has_split):
                cls_folders.append(cls_folder)

        # instantiating datasets for each class
        if activity_type != "all":
            # loading information about activities to be able to filter them
            steps_info_filename = osp.join("/".join(folder.split("/")[:-1]), "steps_info.pickle")
            with open(steps_info_filename, "rb") as handle:
                steps_info = pickle.load(handle)
            # filtering the activities by type
            cls_folders = [f for f in cls_folders if steps_info["cls_to_type"][int(f.split("_")[-2])] == activity_type]

        # instantiating datasets for each class
        self.cls_datasets = [LMDB_Class_Dataset(f, split, transform, truncate) for f in cls_folders]
        self.cls_lens = [len(d) for d in self.cls_datasets]
        self.cls_end_idx = np.cumsum(self.cls_lens)

    def get_step_embedding(self, step_idx):
        for cls_dataset in self.cls_datasets:
            if step_idx in cls_dataset.step_embeddings:
                step_features = cls_dataset.step_embeddings.get(step_idx)
                emb_post = self.transform({"step_features": step_features})["step_features"]
                return emb_post

    def get_step_description(self, step_idx):
        for cls_dataset in self.cls_datasets:
            if step_idx in cls_dataset.step_descriptions:
                return cls_dataset.step_descriptions.get(step_idx)

    def __getitem__(self, idx):
        # find which dataset the idx corresponds to
        dataset_idx = 0
        while idx >= self.cls_end_idx[dataset_idx]:
            dataset_idx += 1

        # find the relative idx within selected dataset
        start_idx = 0 if dataset_idx == 0 else self.cls_end_idx[dataset_idx - 1]
        relative_idx = idx - start_idx
        return self.cls_datasets[dataset_idx][relative_idx]

    def __len__(self):
        return sum(self.cls_lens)


class LMDB_Class_Dataset(Dataset):
    def __init__(self, cls_folder, split="train", transform=None, truncate=0):
        split_name = osp.basename(cls_folder.rstrip("/")).split("_")
        if len(split_name) == 2:
            self.cls_id = int(split_name[0])
        else:
            self.cls_id = int(split_name[1])
        # self.cls_id = int(osp.basename(cls_folder.rstrip('/')).split('_')[1])
        lmdb_filename = osp.basename(cls_folder.rstrip("/")).replace("lmdb", split) + ".lmdb"
        db_path = osp.join(cls_folder, lmdb_filename)
        self.env = lmdb.open(
            db_path, subdir=osp.isdir(db_path), max_readers=1, readonly=True, lock=False, readahead=False, meminit=False
        )
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b"__len__"))
            self.keys = pa.deserialize(txn.get(b"__keys__"))

        if truncate > 0:
            self.length = min(truncate, self.length)

        self.transform = transform

        # reading step embeddings
        dataset_root_dir = "/".join(cls_folder.split("/")[:-3])
        steps_info_filename = osp.join(dataset_root_dir, "steps_info.pickle")
        if osp.exists(steps_info_filename):
            with open(steps_info_filename, "rb") as handle:
                steps_info = pickle.load(handle)
            # take only the embeddings that belong to this class to save memory
            self.step_embeddings = {
                k: v
                for k, v in steps_info["steps_to_embeddings"].items()
                if k in steps_info["cls_to_steps"][self.cls_id]
            }
            self.step_descriptions = {
                k: v
                for k, v in steps_info["steps_to_descriptions"].items()
                if k in steps_info["cls_to_steps"][self.cls_id]
            }

    def __getitem__(self, idx):
        """
        Extracts a sample with the following fields:

        name: name of the example
        cls: class (or activity) of the video
        cls_name: name of the class (or activity)
        num_frames: Number of frames N
        frame_features: features from all N frames in the video, has size [N, d]
        num_steps: Number of steps K
        step_ids: ids of steps that happen in this video, has len K
        step_features: Feature matrix of size [K, d], where d
        step_starts: start of each step (in frames)
        step_ends: end of each step (in frames)
        """
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[idx])
        full_sample_dict = pa.deserialize(byteflow)
        sample_dict = {
            k: v for k, v in full_sample_dict.items() if k in ["name", "cls", "cls_name", "num_steps", "num_subs"]
        }
        sample_dict["frame_features"] = full_sample_dict["frames_features"]
        sample_dict["num_frames"] = np.array(full_sample_dict["frames_features"].shape[0])

        # fill in the dict with step features and their durations

        # OFFSET is the smallest step_id in the datasets (It's not 0 for some datasets)
        step_ids = full_sample_dict["steps_ids"] - CONFIG.DATASET.OFFSET
        # add <eos> token in the end
        sample_dict["step_ids"] = np.append(step_ids, CONFIG.DATASET.NUM_CLASSES)

        # get step features
        if "steps_features" in full_sample_dict.keys():
            sample_dict["step_features"] = full_sample_dict["steps_features"]
        else:
            sample_dict["step_features"] = np.concatenate(
                [self.step_embeddings[k] for k in full_sample_dict["steps_ids"]]
            )

        # transform seconds to steps
        sample_dict["step_starts_sec"] = full_sample_dict["steps_starts"]
        sample_dict["step_starts"] = np.array([Time2FrameNumber(s, 10) // 32 for s in full_sample_dict["steps_starts"]])

        sample_dict["step_ends_sec"] = full_sample_dict["steps_ends"]
        sample_dict["step_ends"] = np.array([Time2FrameNumber(s, 10) // 32 for s in full_sample_dict["steps_ends"]])

        if self.transform is not None:
            sample_dict = self.transform(sample_dict)
        return sample_dict

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"


def subsample(sample, rate=2):
    K = sample["step_features"].shape[0]
    N = sample["frame_features"].shape[0]
    annotations = torch.zeros([K, N], dtype=int)
    for si in range(K):
        start_i = sample["step_starts"][si]
        end_i = sample["step_ends"][si]
        annotations[si, start_i : (end_i + 1)] = sample["step_ids"][si]

    keep = torch.zeros(N, dtype=bool)
    keep[random.randint(0, rate - 1) :: rate] = True
    new_N = keep.to(int).sum()

    # ensuring that we didn't filter out entire steps
    reserved_points = []
    for si in range(K):
        if annotations[si][keep].sum() == 0:
            print("happened")
            points_of_interest = torch.range(0, N - 1)[annotations[si] > 0].cpu()
            step_point = random.randint(points_of_interest.min(), points_of_interest.max())

            # find a point to switch the sample point with
            left_end, right_end = step_point, step_point
            while not keep[left_end] and not keep[right_end]:
                left_end = max(left_end - 1, 0)
                right_end = min(right_end + 1, N - 1)
                while left_end in reserved_points:
                    left_end = max(left_end - 1, 0)
                while right_end in reserved_points:
                    right_end = min(right_end + 1, N - 1)

            swappoint = left_end if keep[left_end] else right_end
            reserved_points.append(swappoint)
            keep[swappoint] = False
            keep[step_point] = True

    # writing filtered out features and gt into sample dict
    step_starts = torch.zeros_like(sample["step_starts"])
    step_ends = torch.zeros_like(sample["step_starts"])
    subsampled_annotations = annotations[:, keep]
    for si in range(K):
        step_scope = torch.range(0, new_N - 1)[subsampled_annotations[si] > 0]
        step_starts[si] = step_scope.min()
        step_ends[si] = step_scope.max()
    sample["step_starts"] = step_starts
    sample["step_ends"] = step_ends
    sample["step_starts_sec"] = step_starts * 3.2 * rate
    sample["step_ends_sec"] = step_ends * 3.2 * rate
    sample["frame_features"] = sample["frame_features"][keep]
    sample["num_frames"] = sample["frame_features"].shape[0]
    return sample
