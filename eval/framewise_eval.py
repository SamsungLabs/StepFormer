import os
import sys
import torch
import numpy as np

from dp.exact_dp import drop_dtw, iou_based_matching
from models.model_utils import compute_sim
from eval.video_segmentation import segment_video_into_slots, segment_video_into_steps
from config import CONFIG


device = "cuda" if torch.cuda.is_available() else "cpu"


def MatchMetrics(frame_assign, sample, fixed_assignment=False):
    step_ids, gt_step_ids = torch.Tensor(sample["step_ids"]), torch.Tensor(sample["gt_step_ids"])
    if fixed_assignment:
        assert len(step_ids) == len(gt_step_ids), "shapes must match in fixed assignment"

    num_frames = sample["frame_features"].shape[0]
    frame_assignment = torch.from_numpy(frame_assign)
    gt_assignment = -torch.ones(num_frames)
    for i in range(len(gt_step_ids)):
        gt_assignment[sample["step_starts"][i] : sample["step_ends"][i] + 1] = i

    if fixed_assignment:
        pred_match_idxs = gt_match_idxs = torch.arange(len(step_ids))
    else:
        if len(step_ids) > 0:
            pred_match_idxs, gt_match_idxs = iou_based_matching(frame_assignment, gt_assignment, step_ids, gt_step_ids)
        else:
            pred_match_idxs = gt_match_idxs = torch.tensor([-1])

    new_pred_seg = torch.from_numpy(frame_assign)
    new_gt_seg = -torch.ones(num_frames)
    intersection, union = 0, 0
    for i, gt_step_id in enumerate(gt_step_ids):
        new_gt_seg[sample["step_starts"][i] : sample["step_ends"][i] + 1] = 999 + i

        # the gt step was not matched to any predicted step
        if i not in gt_match_idxs:
            union += sample["step_ends"][i] - sample["step_starts"][i] + 1
            continue

        match_idx = (gt_match_idxs == i).nonzero(as_tuple=False)[0]
        pred_step_idx = step_ids[pred_match_idxs[match_idx]]
        gt_step_idx = gt_step_ids[gt_match_idxs[match_idx]]
        gt_assignment[sample["step_starts"][i] : sample["step_ends"][i] + 1] = gt_step_idx
        pred = frame_assignment == pred_step_idx
        gt = gt_assignment == gt_step_idx
        intersection += torch.logical_and(pred, gt).to(int).sum()
        union += torch.logical_or(pred, gt).to(int).sum()

        new_pred_seg[pred] = 999 + i
    iou = (intersection / (union + 1e-5)).item()
    acc_all = (new_pred_seg == new_gt_seg).to(float).mean()

    precision, recall = 0, 0
    labeled_mask = new_gt_seg != -1
    if sum(labeled_mask.to(int)) > 0:
        recall = (new_pred_seg[labeled_mask] == new_gt_seg[labeled_mask]).to(float).mean()

    pred_mask = new_pred_seg != -1
    if sum(pred_mask.to(int)) > 0:
        precision = (new_pred_seg[pred_mask] == new_gt_seg[pred_mask]).to(float).mean()
    metrics = {"acc": acc_all, "prec": precision, "rec": recall, "iou": iou}
    return metrics


def evaluate_predicted_steps_zeroshot(
    video_features,
    phrase_features,
    pred_steps,
    json,
    unordered=False,
):
    """Use GT steps to select predicted steps, and then segment the video with remaining predicted steps"""

    # find assignment steps to features
    K = phrase_features.shape[0]
    S = pred_steps.shape[0]
    if K > S:
        pred_steps = torch.cat(pred_steps, torch.zeros_like(pred_steps)[: K - S], dim=0)
    phrase2step_sim = compute_sim(phrase_features, pred_steps, CONFIG.EVAL.L2_NORM)
    ps_costs = -phrase2step_sim.numpy()
    step_drop_costs = np.zeros(S) + ps_costs.max().item() + 1  # drops only when no other choice
    ps_labels = drop_dtw(ps_costs, step_drop_costs, one_to_one=True, return_labels=True).astype(int)
    phrase2step_dict = {phrase_id - 1: step_id for step_id, phrase_id in enumerate(ps_labels) if phrase_id > 0}
    step_features = pred_steps[[phrase2step_dict[i] for i in range(K)]]

    # segment into matched step slots
    gt_sample = pack_gt_sample(video_features, step_features, np.arange(K), json)
    segmentation = segment_video_into_steps(video_features, step_features, unordered)
    metrics = MatchMetrics(segmentation, gt_sample, True)
    return metrics, segmentation


def evaluate_predicted_steps_unsupervised(
    video_features,
    phrase_features,
    pred_steps,
    json,
):
    """Segment the video with predicted steps without using any GT knowledge"""

    # computing matching and drop costs for video-step matching
    segmentation = segment_video_into_slots(video_features, pred_steps)
    present_labels = np.unique(segmentation)
    present_labels = present_labels[present_labels > -1]

    if json is None:
        metrics = None
    else:
        gt_step_ids = np.arange(len(json["phrases"]))
        step_ids = present_labels
        gt_sample = pack_gt_sample(video_features, phrase_features, step_ids, json, gt_step_ids)
        metrics = MatchMetrics(segmentation, gt_sample)
    return metrics, segmentation


def pack_gt_sample(video_features, step_features, step_ids, json, gt_step_ids=None):
    N = video_features.shape[0]
    sample = dict()
    sample["step_ids"] = step_ids
    sample["gt_step_ids"] = step_ids if gt_step_ids is None else gt_step_ids

    # step_features = pred_steps[[phrase2step_dict[pid] for pid in sample["gt_step_ids"]]]
    sample["frame_features"] = video_features
    sample["step_features"] = step_features
    sample["step_starts"] = torch.round(torch.tensor(json["start"])).to(int).clip(0, N - 1)
    sample["step_ends"] = torch.round(torch.tensor(json["end"])).to(int).clip(0, N - 1)
    return sample
