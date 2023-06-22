import os
import torch
import numpy as np
import pandas as pd
import sys
from os import path as osp
import csv
import json
from os.path import join as opj

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from paths import CT_PATH, PROCEL_PATH, ANNOT_PATH


def dict2tensor(d):
    for key, value in d.items():
        if type(value) != str:
            d[key] = torch.from_numpy(np.array(value))
    return d


def Time2FrameNumber(t, ori_fps, fps=10):
    """function to convert segment annotations given in seconds to frame numbers
    input:
        t: is the time (in seconds) that we want to convert to frame number
        ori_fps: is the original fps of the video
        fps: is the fps that we are using to extract frames from the video
    output:
        numf: the frame number corresponding to the time t of a video encoded at fps
    """
    ori2fps_ratio = int(ori_fps / fps)
    ori_numf = t * ori_fps
    numf = int(ori_numf / ori2fps_ratio)
    return numf


def sample_to_device(sample, device):
    for k, v in sample.items():
        try:
            sample[k] = v.to(device)
        except:
            pass
    return sample


def MergeConsec(sample):
    """merge consecutive steps and their corresponding start and end times"""
    if "step_ids" in sample:
        # remove EOS token
        a = sample["step_ids"][:-1]
        eos = sample["step_ids"][-1]

        # create a mask that finds adjacent duplicates
        mask = a[0:-1] != a[1:]
        mask = np.concatenate(([True], mask))
        # filter consec duplicates
        sample["step_ids"] = np.concatenate((a[mask], eos[np.newaxis]))
        sample["step_starts"] = sample["step_starts"][mask]
        sample["step_ends"] = sample["step_ends"][mask]
        sample["step_features"] = sample["step_features"][mask]
        sample["num_steps"] = sample["step_ids"].shape[0]
    return sample


def parse_crosstask_anno(steps_info_path):
    """pasre annotation to txt"""
    with open(steps_info_path, "r") as txtfile:
        reader = csv.reader(txtfile, delimiter=",")
        count = 0
        steps_info = dict()
        for row in reader:
            count += 1
            if count == 1:
                task_id = row[0]
                steps_info[task_id] = {}
            elif count == 4:
                num_steps = row[0]
                steps_info[task_id]["num_steps"] = int(num_steps)
            elif count == 5:
                steps = row
                steps_info[task_id]["steps"] = steps
            elif count >= 6:
                count = 0
    return steps_info


def get_task_steps(task_name, dataset="CrossTask"):
    if dataset == "CrossTask":
        """get task steps based on task name for crosstask"""
        task_file = CT_PATH + "tasks_primary.txt"
        tasks_info = parse_crosstask_anno(task_file)
        task_steps = tasks_info[task_name]["steps"]
    if dataset == "ProceL":
        tasks_info = PROCEL_PATH + "tasks_info.json"
        all_tasks = json.load(open(tasks_info, "r"))
        task_steps = all_tasks[task_name]
    if dataset == "COIN":
        data_folder = "/user/n.dvornik/Git/VisualNarrationProceL-CVPR21/data_coin/"
        annot_folder = os.path.join(data_folder, task_name)
        task_steps = open(os.path.join(annot_folder, "mapping.txt")).read().split("\n")
        task_steps = [s.split(" ")[1].replace("_", " ").lower() for s in task_steps]

    steps_dict = {}
    for i, step in enumerate(task_steps):
        steps_dict[step] = i
    return steps_dict


def get_dataset_info(dataset="CrossTask", mode="val"):
    annot_folder = opj(ANNOT_PATH, "data_" + dataset.lower())
    val_filelist = pd.read_csv(opj(annot_folder, f"{mode}_names.csv"))
    names = [n.replace("-val", "") for n in val_filelist["video_name"].tolist()]
    if dataset == "CrossTask":
        names = ["_".join(n.split("_")[1:]) for n in names]
    tasks = [str(t) for t in val_filelist["task"].tolist()]
    name2task_dict = dict(zip(names, tasks))

    task_steps_dict = dict()
    for task in set(list(name2task_dict.values())):
        task_folder = opj(annot_folder, task)
        task_steps = open(os.path.join(task_folder, "mapping.txt")).read().split("\n")
        task_steps_dict[task] = [s.split(" ")[1].replace("_", " ").lower() for s in task_steps]
    return name2task_dict, task_steps_dict
