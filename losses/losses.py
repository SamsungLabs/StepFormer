import torch
import torch.nn.functional as F
from torch import log, exp
from torchmetrics.functional import accuracy, precision, recall
import numpy as np

from models.model_utils import compute_masked_sims, compute_sim, subsample_videos_with_masks, filter_out_weak_phrases
from dp.soft_dp import batch_drop_dtw_machine, batch_double_drop_dtw_machine
from dp.exact_dp import batch_double_drop_dtw_machine as exact_batch_double_drop_dtw_machine
from dp.exact_dp import batch_drop_dtw_machine as exact_batch_drop_dtw_machine
from dp.exact_dp import fast_batch_double_drop_dtw_machine, batch_NW_machine

# from dp.gpu_nw import gpu_nw
from dp.dp_utils import compute_all_costs, compute_double_costs
from config import CONFIG


def mil_nce(sim, correspondance_mat, gamma, eps=1e-8, hard_ratio=1, mask_out_diag=False, filter_positive=False):
    if filter_positive:
        pos_mask = correspondance_mat.any(1)
        sim, correspondance_mat = sim[pos_mask], correspondance_mat[pos_mask]

    corresp = correspondance_mat.to(torch.float32)
    prod = sim / gamma
    # logsumexp trick happens here
    prod_exp = exp(prod - prod.max(dim=1, keepdim=True).values)
    if mask_out_diag:
        # not taking correspondences on diagonal into consideration
        # only valid if sim is square
        assert prod.shape[0] == prod.shape[1], "sim matrix is not square"
        diag_mask = torch.eye(prod_exp.shape[0], device=sim.device)
        prod_exp = prod_exp * (1 - diag_mask)
    nominator = (prod_exp * corresp).sum(dim=1)
    denominator = prod_exp.sum(dim=1)
    nll = -log(nominator / (denominator + eps))
    if hard_ratio < 1:
        n_hard_examples = int(nll.shape[0] * hard_ratio)
        hard_indices = nll.sort().indices[-n_hard_examples:]
        nll = nll[hard_indices]
    return nll.mean()


def mil_nce_set(sim, correspondance_mat, B, gamma, eps=1e-8):
    prod = sim / gamma
    # logsumexp trick happens here
    prod_exp = exp(prod - prod.max())
    denominator = prod_exp.sum()
    total_loss = 0
    for i in range(B):
        corresp = correspondance_mat == i + 1
        nominator = (prod_exp[corresp]).sum()
        nll = -log(nominator / (denominator + eps))
        total_loss = total_loss + nll.mean()
    return nll / B


def compute_alignment_offline(
    sims,
    keep_percentile,
    top_band_size=0,
    given_droplines=None,
    drop_z=True,
    one_to_many=False,
    many_to_one=False,
    contiguous=False,
):
    # computing alignments (without gradients)
    orig_device = sims[0].device
    # embarisingly, this is faster on CPU than on GPU!
    sims = [s.cpu() for s in sims]
    given_droplines = None if given_droplines is None else [s.cpu() for s in given_droplines]
    with torch.no_grad():
        zx_costs_list = []
        x_drop_costs_list = []
        z_drop_costs_list = []
        for i, sim in enumerate(sims):
            # computing the baseline logit
            top_sim = sim
            if given_droplines is None:
                if top_band_size > 0 and top_band_size < sim.shape[1]:
                    top_sim = sim.topk(top_band_size, dim=1).values

                if keep_percentile > 1:
                    dropline = top_sim.min() - 5
                else:
                    k = max([1, int(torch.numel(top_sim) * keep_percentile)])
                    dropline = torch.topk(top_sim.reshape([-1]), k).values[-1].detach()
            else:
                dropline = given_droplines[i]

            # shift the costs by the drop logits, so I can set drop costs to 0 instead
            zx_costs_list.append(dropline.reshape([1, 1]) - sim)
            z_drop_cost = torch.zeros([sim.size(0)]).to(sim.device)
            x_drop_cost = torch.zeros([sim.size(1)]).to(sim.device)
            z_drop_costs_list.append(z_drop_cost)
            x_drop_costs_list.append(x_drop_cost)

        # TODO figure out if one_to_many and many_to_one should be on
        align_paths, corresp_mats = None, None
        if drop_z:
            if not (one_to_many or many_to_one):
                _, align_paths = batch_NW_machine(zx_costs_list, x_drop_costs_list, z_drop_costs_list)
                # corresp_mats = gpu_nw(zx_costs_list, x_drop_costs_list, z_drop_costs_list)
            else:
                _, align_paths = exact_batch_double_drop_dtw_machine(
                    # _, align_paths = fast_batch_double_drop_dtw_machine(
                    zx_costs_list,
                    x_drop_costs_list,
                    z_drop_costs_list,
                    one_to_many=one_to_many,
                    many_to_one=many_to_one,
                    contiguous=contiguous,
                )
        else:
            _, align_paths = exact_batch_drop_dtw_machine(
                zx_costs_list,
                x_drop_costs_list,
                one_to_many=one_to_many,
                many_to_one=many_to_one,
                contiguous=contiguous,
            )

        if corresp_mats is None:
            corresp_matrices = []
            for b_id, sim in enumerate(sims):
                corresp_matrix = torch.zeros_like(sim)
                for i, j, s in align_paths[b_id]:
                    if s == 0:
                        corresp_matrix[i - 1, j - 1] = 1
                corresp_matrices.append(corresp_matrix.to(orig_device))
                # corresp_matrices.append(corresp_matrix)
    return align_paths, corresp_matrices


def compute_align_loss(
    z_features,
    x_features,
    z_pad_masks,
    x_pad_masks,
    keep_percentile=0.5,
    gamma_zx=10,
    gamma_min=1,
    l2_normalize=False,
    drop_cost_type="logit",
    align_algo="DropDTW",
):
    B = z_features.shape[0]
    zx_costs_list = []
    x_drop_costs_list = []
    z_drop_costs_list = []

    for i in range(B):
        z_feat, x_feat = z_features[i], x_features[i]
        z_feat = z_feat if z_pad_masks is None else z_feat[~z_pad_masks[i]]
        x_feat = x_feat if x_pad_masks is None else x_feat[~x_pad_masks[i]]

        if align_algo == "DropDTW" or align_algo == "MixedDropDTW":
            if align_algo == "MixedDropDTW" and x_feat.shape[0] < z_feat.shape[0]:
                # chosing the shorter sequence as columns
                z_feat, x_feat = x_feat, z_feat
            zx_costs, x_drop_costs, _ = compute_all_costs(
                z_feat, x_feat, gamma_zx, drop_cost_type, keep_percentile, l2_normalize
            )
        elif align_algo == "DoubleDropDTW":
            zx_costs, x_drop_costs, z_drop_costs = compute_double_costs(
                z_feat, x_feat, gamma_zx, drop_cost_type, keep_percentile
            )
            z_drop_costs_list.append(z_drop_costs)

        zx_costs_list.append(zx_costs)
        x_drop_costs_list.append(x_drop_costs)

    # min_costs, _ = batch_double_dropDTW(zx_costs_list, drop_costs_list, gamma_min=gamma_min)
    if align_algo == "DropDTW" or align_algo == "MixedDropDTW":
        min_costs, path_lens = batch_drop_dtw_machine(zx_costs_list, x_drop_costs_list, gamma_min=gamma_min)
        min_costs = min_costs / path_lens
    elif align_algo == "DoubleDropDTW":
        min_costs = batch_double_drop_dtw_machine(
            zx_costs_list, x_drop_costs_list, z_drop_costs_list, gamma_min=gamma_min, exclusive=False
        )

    loss = sum(min_costs) / B
    return loss


def compute_align_corresp_loss(
    z_features,
    x_features,
    z_pad_masks,
    x_pad_masks,
    keep_percentile=0.2,
    top_band=0,
    gamma_zx=10,
    l2_normalize=False,
    given_sims=None,
    given_correspondences=None,
    one_to_many=False,
):
    B = z_features.shape[0]

    # Computing similarities between steps and phrases
    if given_sims is None:
        sims = compute_masked_sims(z_features, x_features, z_pad_masks, x_pad_masks, l2_normalize)
    else:
        sims = given_sims

    if CONFIG.LOSS.PHRASE_CONTRASTIVE_DROP:
        B = z_features.shape[0]
        given_droplines = []
        for i in range(B):
            step = z_features[i]
            negative_idxs = [k for k in range(B) if k != i]
            # negative_idxs = [k for k in range(B) if k != i] if B > 1 else [i]
            negative_phrases = x_features[negative_idxs][~x_pad_masks[negative_idxs]]
            negative_activations = compute_sim(negative_phrases, step, l2_normalize).max(-1).values
            K = int((1 - CONFIG.LOSS.CONTRASTIVE_DROP_PERC) * negative_activations.shape[0])
            given_dropline = torch.topk(negative_activations, K).values[-1].to(step.device)
            given_droplines.append(given_dropline)
    else:
        given_droplines = None

    if given_correspondences is None:
        _, corresp_matrices = compute_alignment_offline(
            sims,
            keep_percentile,
            top_band_size=top_band,
            given_droplines=given_droplines,
            drop_z=True,
            one_to_many=one_to_many,
            many_to_one=False,
            contiguous=False,
        )
    else:
        corresp_matrices = given_correspondences

    # computing loss for every element
    total_loss = 0
    for b_id, sim in enumerate(sims):
        # computing the correspondence matrix
        corresp_matrix = corresp_matrices[b_id]
        good_row_mask = (corresp_matrix == 1).any(1)
        good_col_mask = (corresp_matrix == 1).any(0)

        # softmax along remaining rows = competition between columns
        if good_row_mask.any():
            row_sim = sim[good_row_mask, :]
            row_corresp = corresp_matrix[good_row_mask, :]
            row_loss = mil_nce(row_sim, row_corresp, gamma_zx)
            total_loss = total_loss + row_loss

        # softmax along remaining columns = competition between rows
        if good_col_mask.any():
            col_sim = sim[:, good_col_mask]
            col_corresp = corresp_matrix[:, good_col_mask]
            col_loss = mil_nce(col_sim.T, col_corresp.T, gamma_zx)
            total_loss = total_loss + col_loss

    total_loss /= B
    return total_loss, sims, corresp_matrices


def compute_contrastive_loss(
    steps,
    phrases,
    steps_pad_mask,
    phrases_pad_mask,
    gamma_zx,
    l2_normalize,
    set_to_set=False,
    contrast_half=False,
):
    B = len(steps)

    # handling presence masks
    if steps_pad_mask is None:
        steps_pad_mask = torch.zeros(steps.shape[:2]).to(steps.device)
    steps_mask = ~(steps_pad_mask.to(bool))

    if phrases_pad_mask is None:
        phrases_pad_mask = torch.zeros(phrases.shape[:2]).to(phrases.device)
    phrases_mask = ~(phrases_pad_mask.to(bool))

    # computing similarities
    all_steps = steps[steps_mask]
    all_phrases = phrases[phrases_mask]
    sim = compute_sim(all_steps, all_phrases, l2_normalize)

    # computing the correspondence matrices for every batch element
    corresp_mats = torch.zeros(all_steps.size(0), all_phrases.size(0)).to(steps.device)
    k_start, n_start = 0, 0
    for b in range(B):
        k_end = steps_mask[b].to(int).sum() + k_start
        n_end = phrases_mask[b].to(int).sum() + n_start
        corresp_mats[k_start:k_end, n_start:n_end] = b + 1
        k_start, n_start = k_end, n_end
    assert k_end == all_steps.size(0) and n_end == all_phrases.size(0), "Something is not right"

    if set_to_set:
        total_loss = mil_nce_set(sim, corresp_mats, B, gamma_zx * 3)
    else:
        corresp_mat = corresp_mats > 0
        # softmax along remaining rows = competition between phrases
        row_loss = mil_nce(sim, corresp_mat, gamma_zx)
        total_loss = row_loss
        if not contrast_half:
            # softmax along remaining columns = competition between steps
            col_loss = mil_nce(sim.T, corresp_mat.T, gamma_zx)
            total_loss = total_loss + col_loss

    return total_loss


def compute_cls_head_loss(steps_cls, phrase_corresp_mats):
    """compute BCE loss for the classification head"""
    BS = len(phrase_corresp_mats)
    targets = []
    for i in range(BS):
        tgt = torch.sum(phrase_corresp_mats[i], dim=1)
        targets.append(tgt)

    targets = torch.stack(targets)
    p_weight = (targets == 0).sum() / max(1, (targets == 1).sum())
    steps_cls = steps_cls.squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(steps_cls, targets, pos_weight=p_weight)
    probs = torch.sigmoid(steps_cls)
    preds = (probs > 0.5).to(float).flatten()
    acc = accuracy(preds, targets.flatten().to(int))
    prec = precision(preds, targets.flatten().to(int))
    rec = recall(preds, targets.flatten().to(int))

    return loss, prec, rec, acc


def compute_video_step_seg_loss(
    videos,
    texts,
    steps,
    video_pad_masks,
    text_pad_masks,
    keep_percentile=0.1,
    top_band=0,
    gamma_zx=10,
    l2_normalize=False,
    contrast_level="video",
    contrast_frames=True,
    contrast_steps=False,
    max_len_after_subsample=300,
    keep_best_phrases=100,
    subsample_mode="mean",
):
    assert contrast_level in ["video", "batch_all", "batch_pos"], f"Impossible contrast_level: {contrast_level}"
    B = videos.shape[0]

    # reduce the number of frames and phrases
    videos, video_pad_masks = subsample_videos_with_masks(
        videos, video_pad_masks, max_len_after_subsample, subsample_mode
    )
    tv_sims = compute_masked_sims(texts, videos, text_pad_masks, video_pad_masks, l2_normalize)
    texts, text_pad_masks, tv_sims = filter_out_weak_phrases(
        texts, text_pad_masks, tv_sims, max_phrases=keep_best_phrases
    )

    # Computing alignment between phrases and video
    _, tv_corresp_matrices = compute_alignment_offline(
        tv_sims,
        keep_percentile,
        top_band_size=top_band,
        drop_z=True,
        one_to_many=True,
        many_to_one=False,
        contiguous=True,
    )

    video_segmentations = []
    all_kept_phrase_ids, all_labels = [], []
    ps_sims = []
    phrase_offset = 0
    for i, corresp_mat in enumerate(tv_corresp_matrices):
        # compute video segmentation
        phrase_ids, video_ids = corresp_mat.nonzero(as_tuple=True)
        labels = phrase_ids + phrase_offset
        video_segmentation = -i - torch.ones(corresp_mat.shape[1]).to(int).to(labels.device)
        video_segmentation[video_ids] = labels
        video_segmentations.append(video_segmentation)

        # compute kept steps
        kept_phrase_ids = phrase_ids.unique()
        kept_phrases = texts[i][kept_phrase_ids]
        ps_sims.append(compute_sim(kept_phrases, steps[i], l2_normalize))

        # put the necessary stuff into lists
        all_kept_phrase_ids.append(kept_phrase_ids)
        all_labels.append(labels.unique())
        phrase_offset += kept_phrase_ids.max() + 1

    # computing 1-to-1 alignment between kept phrases and steps
    do_drop_z = any([len(all_kept_phrase_ids[i]) > steps.shape[1] for i in range(B)])
    _, ps_corresp_matrices = compute_alignment_offline(
        ps_sims,
        keep_percentile=2,
        top_band_size=0,
        drop_z=do_drop_z,
        one_to_many=False,
        many_to_one=False,
        contiguous=False,
    )

    # creating phrase to video labels
    kept_step_ids, all_kept_steps, all_kept_step_labels = [], [], []
    step_segmentations = []
    for i, corresp_mat in enumerate(ps_corresp_matrices):
        kept_phrase_idxs, step_idxs = corresp_mat.nonzero(as_tuple=True)
        step_labels = all_labels[i][kept_phrase_idxs]
        step_segmentation = -i - torch.ones(corresp_mat.shape[1]).to(int).to(step_labels.device)
        step_segmentation[step_idxs] = step_labels
        step_segmentations.append(step_segmentation)

        kept_step_ids.append(step_idxs)
        all_kept_steps.append(steps[i][step_idxs])
        all_kept_step_labels.append(step_labels)

    if contrast_level == "video":
        full_loss = 0
        for i in range(B):
            video = videos[i][~video_pad_masks[i]]
            kept_steps = all_kept_steps[i]
            sv_sim = compute_sim(kept_steps, video, l2_normalize)
            corresp_mat = all_kept_step_labels[i][:, None] == video_segmentations[i][None, :]

            if contrast_frames:
                full_loss += mil_nce(sv_sim, corresp_mat, gamma_zx)
            if contrast_steps:
                full_loss += mil_nce(sv_sim.T, corresp_mat.T, gamma_zx, filter_positive=True)
        full_loss = full_loss / B
    elif contrast_level in ["batch_all", "batch_pos"]:
        # concat all used steps and their labels
        if contrast_level == "batch_all":
            all_used_steps = torch.cat(steps.unbind(0), 0)
            all_step_labels = torch.cat(step_segmentations, 0)
        else:
            all_used_steps = torch.cat(all_kept_steps, 0)
            all_step_labels = torch.cat(all_kept_step_labels, 0)

        # concat all used video frames and their labels
        all_used_frames = torch.cat([v[~video_pad_masks[i]] for i, v in enumerate(videos)], 0)
        all_frame_labels = torch.cat(video_segmentations, 0)

        # define similarities, corresp_matrices, and compute the loss
        batch_sv_sim = compute_sim(all_used_steps, all_used_frames, l2_normalize)
        batch_corresp_mat = all_step_labels[:, None] == all_frame_labels[None, :]

        full_loss = 0
        if contrast_frames:
            full_loss += mil_nce(batch_sv_sim, batch_corresp_mat, gamma_zx, filter_positive=True)
        if contrast_steps:
            full_loss += mil_nce(batch_sv_sim.T, batch_corresp_mat.T, gamma_zx, filter_positive=True)

    return full_loss, ps_sims, ps_corresp_matrices


def compute_step_attn_reg_loss(steps, videos, video_pad_masks, gamma_zx=10, l2_normalize=False):
    """Enforces attention of different steps in the video to have picks at different locations"""
    B = steps.shape[0]
    sv_sims = compute_masked_sims(steps, videos, None, video_pad_masks, l2_normalize)

    loss = 0
    for i, sv_sim in enumerate(sv_sims):
        sv_probs = F.softmax(sv_sim / gamma_zx, dim=1)
        prob_sim = compute_sim(sv_probs, sv_probs, l2_normalize)
        # prob_sim = 1 - (sv_probs[:, None, :] - sv_probs[None, :, :]).norm(dim=2, p=2)
        K = prob_sim.shape[0]
        prob_sim = prob_sim * (1 - torch.eye(K, device=prob_sim.device))
        loss = loss + prob_sim.sum() / (K * (K - 1) / 2)
    return loss / B


def compute_trans_step_video_reg_loss(steps, steps_trans, gamma_zx=10, l2_normalize=False):
    B = steps.shape[0]
    st_s_sims = compute_masked_sims(steps, steps_trans, None, None, l2_normalize)

    total_loss = 0
    for i, sim in enumerate(st_s_sims):
        loss = mil_nce(sim, torch.eye(len(sim)).to(sim.device), gamma=gamma_zx)
        total_loss = total_loss + loss
    return total_loss
