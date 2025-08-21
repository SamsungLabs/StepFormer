import torch
import torch.nn.functional as F
import numpy as np
import math

from losses.losses import mil_nce, compute_alignment_offline
from models.model_utils import compute_masked_sims, compute_sim, subsample_video, filter_out_weak_phrases
from config import CONFIG


def compute_video_align_corresp_loss(
    z_features,
    x_features,
    z_pad_masks,
    x_pad_masks,
    keep_percentile=0.2,
    gamma_zx=10,
    l2_normalize=False,
    max_len_after_subsample=300,
    contrast_frames=True,
):
    """z_feat is query steps, x_feat - video features"""
    B = len(z_features)

    # Computing similarities between steps and phrases
    sims = []
    for i in range(B):
        z_feat, x_feat = z_features[i], x_features[i]
        steps = z_feat if z_pad_masks is None else z_feat[~z_pad_masks[i]]
        video = x_feat if x_pad_masks is None else x_feat[~x_pad_masks[i]]
        video = subsample_video(video, max_len_after_subsample)
        if l2_normalize:
            sims.append(F.normalize(steps, dim=1) @ F.normalize(video, dim=1).T)
        else:
            sims.append(steps @ video.T)

    _, corresp_matrices = compute_alignment_offline(
        sims, keep_percentile, drop_z=False, one_to_many=True, many_to_one=False, contiguous=True
    )

    # computing loss for every element
    total_loss = 0
    for b_id, sim in enumerate(sims):
        # computing the correspondence matrix
        corresp_matrix = corresp_matrices[b_id]
        good_row_mask = (corresp_matrix == 1).any(1)
        good_col_mask = (corresp_matrix == 1).any(0)

        if contrast_frames:
            # softmax along remaining rows = competition between frames
            row_sim = sim[good_row_mask, :]
            row_corresp = corresp_matrix[good_row_mask, :]
            row_loss = mil_nce(row_sim, row_corresp, gamma_zx)
        else:
            row_loss = 0

        # softmax along remaining columns = competition between steps
        col_sim = sim[:, good_col_mask]
        col_corresp = corresp_matrix[:, good_col_mask]
        col_loss = mil_nce(col_sim.T, col_corresp.T, gamma_zx)

        total_loss += row_loss + col_loss

    total_loss /= B
    return total_loss, sims, corresp_matrices


def compute_npair_video_reg_loss(
    z_features,
    x_features,
    z_pad_masks,
    x_pad_masks,
    gamma_zx=10,
    l2_normalize=False,
    max_len_after_subsample=1000,
):
    """z_feat is query steps, x_feat - video features"""
    B = len(z_features)

    # Computing similarities between steps and phrases
    loss = 0
    for i in range(B):
        z_feat, x_feat = z_features[i], x_features[i]
        steps = z_feat if z_pad_masks is None else z_feat[~z_pad_masks[i]]
        video = x_feat if x_pad_masks is None else x_feat[~x_pad_masks[i]]
        if max_len_after_subsample < CONFIG.DATASET.MAX_VIDEO_LEN:
            video = subsample_video(video, max_len_after_subsample)

        sv_sims = compute_sim(steps, video, l2_normalize)
        # softmaxing the similarities
        sv_probs = F.softmax(sv_sims / gamma_zx, dim=0)  # [K, N]

        # go from relative ratios to video points
        N = len(video)
        N_sample = len(steps)
        N_pos = math.ceil(N / N_sample)

        # sampling random points in the video for comparison
        sampled_points = torch.linspace(1 / (N_sample + 1), N_sample / (N_sample + 1), N_sample)
        sampled_idxs = (sampled_points * N).to(int).unique().to(video.device)
        N_sample = len(sampled_idxs)  # update because of the unique() operator

        # sample 3 positives for each sampled point point
        pos_offsets = torch.randint(-N_pos, N_pos + 1, (N_sample, 3), device=video.device)
        pos_sampled_idxs = (sampled_idxs[:, None] + pos_offsets).reshape(-1)
        # cleanup sampled idxs outside of video range
        pos_sampled_idxs = pos_sampled_idxs[torch.logical_and(pos_sampled_idxs < N, pos_sampled_idxs > 0)]
        # merge all the sampled idxs together
        sampled_idxs = torch.cat([sampled_idxs, pos_sampled_idxs]).unique()

        # find positive and negative relationships between sampled idxs
        relative_positions = (sampled_idxs[:, None] - sampled_idxs[None, :]).abs()
        pos_corresps = relative_positions <= N_pos

        # keep only samples that have at least one positive
        has_pos_match_mask = pos_corresps.to(int).sum(1) > 1
        sampled_idxs = sampled_idxs[has_pos_match_mask]
        pos_corresps = pos_corresps[has_pos_match_mask]

        # computing l2 distance between step probability vectors, for each pair of frames
        vv_dist = ((sv_probs[:, sampled_idxs, None] - sv_probs[:, None, sampled_idxs]) ** 2).sum(0)  # [N, N]
        # computing the loss per sample
        loss = loss + mil_nce(-vv_dist, pos_corresps, gamma=0.05, mask_out_diag=True)
    return loss / B


def compute_intra_video_loss(
    z_features,
    x_features,
    z_pad_masks,
    x_pad_masks,
    gamma_zx=10,
    l2_normalize=False,
    max_len_after_subsample=800,
    pos_neighborhood=0.1,
    neg_neighborhood=0.35,
    hard_negative_ratio=0.2,
    sample_points_perc=0.05,
):
    """z_feat is query steps, x_feat - video features"""
    B = len(z_features)

    # Computing similarities between steps and phrases
    pos_loss, neg_loss = 0, 0
    for i in range(B):
        z_feat, x_feat = z_features[i], x_features[i]
        steps = z_feat if z_pad_masks is None else z_feat[~z_pad_masks[i]]
        video = x_feat if x_pad_masks is None else x_feat[~x_pad_masks[i]]

        subsample_rate = np.ceil(video.size(0) / max_len_after_subsample).astype(int)
        if subsample_rate > 1:
            video = video[::subsample_rate]

        if l2_normalize:
            sv_sims = F.normalize(steps, dim=1) @ F.normalize(video, dim=1).T
        else:
            sv_sims = steps @ video.T
        # softmaxing the similarities
        sv_probs = F.softmax(sv_sims / gamma_zx, dim=0)  # [K, N]

        # go from relative ratios to video points
        N = sv_probs.shape[1]
        N_pos = max(int(pos_neighborhood * N), 1)
        N_neg = max(int(neg_neighborhood * N), 1)
        N_sample = max(int(sample_points_perc * N), 2)
        # sampling random points in the video for comparison
        sampled_idxs = torch.randint(0, N, (N_sample,), device=sv_probs.device).unique()
        N_sample = len(sampled_idxs)  #  update because of the unique() operator
        # sample more points that are known to be have positives
        pos_offsets = torch.randint(1, N_pos + 1, (N_sample,), device=sv_probs.device)
        pos_sampled_idxs = sampled_idxs + pos_offsets
        pos_sampled_idxs = pos_sampled_idxs[pos_sampled_idxs < N]
        sampled_idxs = torch.cat([sampled_idxs, pos_sampled_idxs])

        # find positive and negative relationships between sampled idxs
        relative_positions = (sampled_idxs[:, None] - sampled_idxs[None, :]).abs()
        pos_mask = relative_positions <= N_pos
        neg_mask = relative_positions >= N_neg

        # computing l2 distance between step probability vectors, for each pair of frames
        vv_dist = ((sv_probs[:, sampled_idxs, None] - sv_probs[:, None, sampled_idxs]) ** 2).sum(0)  # [N, N]
        # computing the loss per sample
        pos_filter, neg_filter = pos_mask.to(bool).any(1), neg_mask.to(bool).any(1)
        if pos_filter.any():
            pos_mask = pos_mask[pos_filter].to(float)
            pos_loss = pos_loss + ((vv_dist[pos_filter] * pos_mask).sum(1) / pos_mask.sum(1)).mean()
        if neg_filter.any():
            neg_mask = neg_mask[neg_filter].to(float)
            neg_loss = neg_loss - ((vv_dist[neg_filter] * neg_mask).sum(1) / neg_mask.sum(1)).mean()

    intra_video_loss = (pos_loss + neg_loss) / B
    return intra_video_loss
