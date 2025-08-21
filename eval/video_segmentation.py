import torch
import numpy as np

from dp.exact_dp import drop_dtw, double_drop_dtw, iou_based_matching
from models.model_utils import compute_sim
from config import CONFIG


def segment_video_into_steps(frame_features, step_features, unordered):
    if step_features.shape[0] == 0:
        return -np.ones(frame_features.shape[0])

    sim = compute_sim(step_features, frame_features, CONFIG.EVAL.L2_NORM).cpu()
    frame_features, step_features = frame_features.cpu(), step_features.cpu()

    k = max([1, int(torch.numel(sim) * CONFIG.EVAL.KEEP_PERCENTILE)])
    baseline_logit = torch.topk(sim.reshape([-1]), k).values[-1].detach()
    baseline_logits = baseline_logit.repeat([1, sim.shape[1]])[0]  # making it of shape [1, N]
    zx_costs, drop_costs = -sim, -baseline_logits
    zx_costs, drop_costs = [t.detach().cpu().numpy() for t in [zx_costs, drop_costs]]
    sim = sim.detach().cpu().numpy()

    if unordered:
        max_vals, optimal_assignment = np.max(sim, axis=0), np.argmax(sim, axis=0)
        optimal_assignment[max_vals < baseline_logit.item()] = -1
    else:
        optimal_assignment = drop_dtw(zx_costs, drop_costs, return_labels=True) - 1
    return optimal_assignment


def segment_video_into_slots(video_features, pred_steps):
    sim = compute_sim(pred_steps, video_features, l2_norm=CONFIG.EVAL.L2_NORM).detach()
    if CONFIG.EVAL.FIXED_DROP_SIM == -1:
        k = max([1, int(torch.numel(sim) * CONFIG.EVAL.KEEP_PERCENTILE)])
        baseline_logit = torch.topk(sim.reshape([-1]), k).values[-1].detach()
    else:
        baseline_logit = torch.tensor(CONFIG.EVAL.FIXED_DROP_SIM)
    baseline_logits = baseline_logit.repeat([1, sim.shape[1]])  # making it of shape [1, N]
    x_drop_costs = -baseline_logits.squeeze()
    zx_costs = -sim

    z_drop_costs = -baseline_logit.repeat([1, sim.shape[0]]).squeeze()
    zx_costs = zx_costs - z_drop_costs[0].reshape([1, 1])
    z_drop_costs = z_drop_costs - z_drop_costs[0]
    x_drop_costs = x_drop_costs - x_drop_costs[0]
    segmentation = double_drop_dtw(zx_costs.numpy(), x_drop_costs.numpy(), z_drop_costs.numpy(), return_labels=True) - 1
    return segmentation
