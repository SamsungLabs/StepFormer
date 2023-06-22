import os
import torch
import scipy.optimize
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F
from glob import glob
from paths import ANNOT_PATH

from eval.video_segmentation import segment_video_into_slots
from eval.annot_utils import get_label_list, get_frame_pred_list


def eval_unsup_segmentation(dataset, task, pred_dir, n_frames=16, framerate=16, mode="val", verbose=False):
    # data folder
    task = task if mode == "train" else task + "-val"
    data_dir = os.path.join(ANNOT_PATH, "data_" + dataset.lower())
    task_pred_dir = os.path.join(pred_dir, task)

    # find files for evaluation
    pred_files = sorted(glob(os.path.join(task_pred_dir, "*.txt")))
    pred_list = [np.loadtxt(f, dtype=int) for f in pred_files]
    video_lens = [len(p) for p in pred_list]
    fid_list = [f.split("/")[-1].split(".")[0] for f in pred_files]

    # the duration and fps of videos (can be given if known, otherwise will be estimated by n_frames and framerate)
    dur_list = [video_len * n_frames / framerate for video_len in video_lens]
    fps_list = [framerate for i in video_lens]

    pred_list = get_frame_pred_list(pred_list, dur_list, fps_list, t_segment=n_frames / framerate)
    annot_dir = os.path.join(data_dir, task, "annotations")
    label_list = get_label_list(annot_dir, fid_list, dur_list, fps_list)
    metric = global_framewise_eval(pred_list, label_list)
    res_string = "Task: {}, Precision: {:.2%}, Recall: {:.2%}, F1-score: {:.2%}, " "MoF: {:.2%}, MoF-bg: {:.2%}".format(
        task, *metric
    )
    if verbose:
        print(res_string)
    return metric


def global_framewise_eval(pred_list, label_list):
    preds = np.concatenate(pred_list)
    labels = np.concatenate(label_list)

    k_pred = int(preds.max()) + 1
    k_label = int(labels.max()) + 1

    overlap = np.zeros([k_pred, k_label])
    for i in range(k_pred):
        for j in range(k_label):
            overlap[i, j] = np.sum((preds == i) * (labels == j))
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-overlap / preds.shape[0])
    K = max(k_pred, k_label)

    bg_row_ind = np.concatenate([row_ind, -np.ones(K + 1 - row_ind.shape[0], dtype=np.int32)])
    bg_col_ind = np.concatenate([col_ind, -np.ones(K + 1 - col_ind.shape[0], dtype=np.int32)])
    acc = np.mean(bg_col_ind[preds] == bg_row_ind[labels])
    acc_steps = np.mean(bg_col_ind[preds[labels >= 0]] == bg_row_ind[labels[labels >= 0]])

    results = []
    for i, p in enumerate(row_ind):
        correct = preds[labels == col_ind[i]] == p
        if correct.shape[0] == 0:
            num_correct = 0
        else:
            num_correct = np.sum(correct)
        num_label = np.sum(labels == col_ind[i])
        num_pred = np.sum(preds == p)
        results.append([num_correct, num_label, num_pred])

    for i in range(k_pred):
        if i not in row_ind:
            num_correct = 0
            num_label = 0
            num_pred = np.sum(preds == i)
            results.append([num_correct, num_label, num_pred])

    for j in range(k_label):
        if j not in col_ind:
            num_correct = 0
            num_label = np.sum(labels == j)
            num_pred = 0
            results.append([num_correct, num_label, num_pred])

    results = np.array(results)

    precision = np.sum(results[:, 0]) / (np.sum(results[:, 2]) + 1e-10)
    recall = np.sum(results[:, 0]) / (np.sum(results[:, 1]) + 1e-10)
    fscore = 2 * precision * recall / (precision + recall + 1e-10)

    return [precision, recall, fscore, acc, acc_steps]


def video_segmentation(x, K, bg_ratio):
    """cluster video features directly (x is a video)"""
    kmeans = KMeans(n_clusters=K, random_state=None).fit(x)
    preds = kmeans.labels_
    centers = kmeans.cluster_centers_
    preds = np.array([label for label in kmeans.labels_])
    for k in range(K):
        pos = np.where(preds == k)[0]
        dist = np.sum((x[pos] - centers[k]) ** 2, axis=-1)
        tmp = preds[pos]
        # predict segments as background if the distance from cluster center is large
        if dist.shape[0] > 0:
            threshold = np.sort(dist)[int((1 - bg_ratio) * dist.shape[0]) - 1]
            tmp[dist > threshold] = -1
        preds[pos] = tmp
    return preds


def filter_clusters(points, centers, labels, bg_ratio, dist_metric):
    new_labels = np.array(labels)
    all_dists = pairwise_distances(points, centers, metric=dist_metric)
    for k in range(len(centers)):
        cluster_point_idxs = np.where(labels == k)[0]
        N = len(cluster_point_idxs)
        cluster_dists = all_dists[cluster_point_idxs, k]
        outlier_threshold = np.sort(cluster_dists)[int(N * (1 - bg_ratio))]
        drop_idxs = np.where(cluster_dists >= outlier_threshold)[0]
        new_labels[cluster_point_idxs[drop_idxs]] = -1
    return new_labels


def step_video_segmentation(all_vids, all_steps, K, bg_ratio=0, keep_percentile=0.1, dist_metric="cosine"):
    """cluster decoded steps across all videos and assign video frames to each step cluster"""

    # align videos with steps usign DoubleDropDTW
    N_steps_total = 0
    all_kept_steps = []
    all_global_labels = []
    for video, steps in zip(all_vids, all_steps):
        step_labels = segment_video_into_slots(video, steps)
        kept_step_idxs = np.unique(step_labels[step_labels > -1])

        # create globally unique step labels
        new_step_labels = -np.ones_like(step_labels, dtype=np.int32)
        for new_id, old_id in enumerate(np.unique(step_labels[step_labels > -1])):
            new_step_labels[step_labels == old_id] = new_id + N_steps_total
        all_global_labels.append(new_step_labels)

        kept_steps_vid = []
        for idx in kept_step_idxs:
            kept_steps_vid.append(video[step_labels == idx].mean(0))
        kept_steps_vid = torch.stack(kept_steps_vid, 0)
        all_kept_steps.append(kept_steps_vid)

        N_steps_total += len(kept_step_idxs)

    all_kept_steps = torch.cat(all_kept_steps, dim=0)

    # Cluster remaining steps
    kmeans = KMeans(n_clusters=K, random_state=None).fit(all_kept_steps)
    preds = kmeans.labels_
    centers = kmeans.cluster_centers_

    # creating step_to_cluster mapping
    if bg_ratio > 0:
        # in case we still want to filter out some steps
        preds = filter_clusters(all_kept_steps, centers, preds, bg_ratio, dist_metric)
    step_id_to_cluster_id = dict(enumerate(preds))

    # mapping step labels into cluster labels, for every video
    all_preds_labels = []
    for i, global_labels in enumerate(all_global_labels):
        pred_labels = -np.ones_like(global_labels, dtype=np.int32)
        for step_id in np.unique(global_labels):
            if step_id > -1:
                pred_labels[global_labels == step_id] = step_id_to_cluster_id[step_id]
        all_preds_labels.append(pred_labels)

    return all_preds_labels, centers
