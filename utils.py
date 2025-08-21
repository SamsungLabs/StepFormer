import os
import yaml
import torch
from glob import glob
import numpy as np
import logging


def get_logger(log_file=None, level="INFO", to_console=True):
    logger = logging.getLogger("logger")
    logger.setLevel(level)
    # logging to file
    if log_file is not None:
        logger.addHandler(logging.FileHandler(log_file))
    # logging to console
    if to_console:
        logger.addHandler(logging.StreamHandler())
    return logger


def get_world_info():
    size, rank = 1, 0
    try:
        import horovod.torch as hvd

        size, rank = hvd.size(), hvd.rank()
    except:
        if torch.distributed.is_initialized():
            size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
    return size, rank


def log_info(message, logger=None):
    printer = print if logger is None else logger.info
    if get_world_info()[1] == 0:
        printer(message)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def load_yaml(filename):
    with open(filename, "r") as stream:
        try:
            dict = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return dict


def dump_yaml(filename, dict):
    with open(filename, "w") as file:
        try:
            yaml.safe_dump(dict, file, default_flow_style=False)
        except yaml.YAMLError as exc:
            print(exc)
    return dict


def Time2FrameNumber(t, ori_fps, fps=10):
    """function to convert segment annotations given in seconds to frame numbers
    input:
        ori_fps: is the original fps of the video
        fps: is the fps that we are using to extract frames from the video
        num_frames: is the number of frames in the video (under fps)
        t: is the time (in seconds) that we want to convert to frame number
    output:
        numf: the frame number corresponding to the time t of a video encoded at fps
    """
    ori2fps_ratio = int(ori_fps / fps)
    ori_numf = t * ori_fps
    numf = int(ori_numf / ori2fps_ratio)
    return numf


def RemoveDuplicates(a):
    """function to remove duplicate steps"""
    filtered = []
    keep_ids = []
    nums = a.shape[0]
    for i in range(nums):
        if a[i] in filtered:
            continue
        else:
            filtered.append(a[i])
            keep_ids.append(i)
    filtered = torch.stack(filtered)
    keep_ids = torch.tensor(keep_ids)
    return filtered, keep_ids


def VidList2Batch(samples, VID_LEN=224):
    """create a batch of videos of the same size from input sequences"""
    # create data needed for training
    vids = []
    batch_size = len(samples)
    for b in range(batch_size):
        numf = samples[b]["frame_features"].shape[0]
        unpadded_vid = samples[b]["frame_features"].T
        # if video is shorter than desired length ==> PAD
        if numf < VID_LEN:
            pad = torch.nn.ConstantPad1d((0, VID_LEN - numf), 0)
            vids.append(pad(unpadded_vid))
        # if video is longer than desired length ==> STRIDED SAMPLING
        elif numf > VID_LEN:
            stride = int(numf // VID_LEN)
            pad = unpadded_vid[:, ::stride]
            vids.append(pad[:, :VID_LEN])
        else:
            pad = unpadded_vid
            vids.append(pad)
    vids = torch.stack(vids, dim=0)

    return vids


def Steps2Batch(steps, num_steps):
    """create a list of lists of the steps"""
    st = 0
    batched_steps = []
    for i in range(len(num_steps)):
        ed = st + num_steps[i]
        batched_steps.append(steps[st:ed, :])
        st = ed
    return batched_steps
