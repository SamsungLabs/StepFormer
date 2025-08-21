import os
import torch
import transformers
import numpy as np
import torch.nn.functional as F
from glob import glob

from config import CONFIG
from models.video_transformer import VideoTransformerQueryDecoder
from paths import PROJECT_PATH


def compute_sim(z, x, l2_norm):
    if l2_norm:
        return F.normalize(z, dim=1) @ F.normalize(x, dim=1).T
    else:
        return z @ x.T


def cosine_sim(x, z):
    cos_sim_fn = torch.nn.CosineSimilarity(dim=1)
    return cos_sim_fn(x[..., None], z.T[None, ...])


def cos_dist(x, z):
    cos_sim_fn = torch.nn.CosineSimilarity(dim=1)
    return (1 - cos_sim_fn(x[..., None], z.T[None, ...])) / 2


def l2_dist(x, z):
    dist_squared = (x**2).sum() + (z**2).sum() - 2 * x @ z.T
    return torch.clamp(dist_squared, min=0).sqrt()


def cos_loglikelihood(x, z, gamma=0.1, z_dim=1):
    cos_sim = cosine_sim(x, z)
    probs = F.softmax(cos_sim / gamma, dim=z_dim)
    return torch.log(probs)


def unique_softmax(sim, labels, gamma=1, dim=0):
    assert sim.shape[0] == labels.shape[0]
    labels = labels.detach().cpu().numpy()
    _, unique_index, unique_inverse_index = np.unique(labels, return_index=True, return_inverse=True)
    unique_sim = sim[unique_index]
    unique_softmax_sim = torch.nn.functional.softmax(unique_sim / gamma, dim=dim)
    softmax_sim = unique_softmax_sim[unique_inverse_index]
    return softmax_sim


def compute_masked_sims(z, x, z_pad_mask, x_pad_mask, l2_normalize=False, softmax_dim=None, gamma=None):
    # z ~ [B, K, d], x ~ [B, N, d]
    if l2_normalize:
        z, x = F.normalize(z, dim=-1), F.normalize(x, dim=-1)
    pad_sims = torch.einsum("bkd,bnd->bkn", z, x)
    masked_sims = []
    for i in range(x.shape[0]):
        masked_sim = pad_sims[i]
        masked_sim = masked_sim if z_pad_mask is None else masked_sim[~z_pad_mask[i], :]
        masked_sim = masked_sim if x_pad_mask is None else masked_sim[:, ~x_pad_mask[i]]
        if softmax_dim is not None:
            masked_sim = F.softmax(masked_sim / gamma, dim=softmax_dim)
        masked_sims.append(masked_sim)
    return masked_sims


def subsample_video(video, max_len_after_subsample, strategy="select", force_subsample=False, json=None):
    vid_len = video.shape[0]
    subsample_rate = np.ceil(vid_len / max_len_after_subsample).astype(int)
    if subsample_rate > 1 or force_subsample:
        if strategy == "select":
            video = video[::subsample_rate]
            if json is not None:
                raise NotImplementedError
        elif strategy == "mean":
            if isinstance(video, np.ndarray):
                video = torch.from_numpy(video).to(float)
                video = F.adaptive_avg_pool1d(video.t()[None, :], max_len_after_subsample)[0].t()
                video = video.numpy()
            else:
                video = F.adaptive_avg_pool1d(video.t()[None, :], max_len_after_subsample)[0].t()
            if json is not None:
                scale = max_len_after_subsample / vid_len
                json["start"] = [t * scale for t in json["start"]]
                json["end"] = [t * scale for t in json["end"]]
                return video, json
    return video


def subsample_videos_with_masks(videos, video_pad_masks, max_len_after_subsample, strategy="select"):
    new_videos = []
    new_max_len = 0
    for i in range(videos.shape[0]):
        video = videos[i][~video_pad_masks[i]]
        video = subsample_video(video, max_len_after_subsample, strategy)
        new_videos.append(video)
        new_max_len = max(new_max_len, len(video))

    padded_videos, padded_masks = [], []
    for new_video in new_videos:
        new_len = len(new_video)

        padded_video = F.pad(new_video, [0, 0, 0, new_max_len - new_len])
        padded_videos.append(padded_video)

        pad_mask = torch.ones(new_max_len).to(bool)
        pad_mask[:new_len] = False
        padded_masks.append(pad_mask)

    return torch.stack(padded_videos, 0), torch.stack(padded_masks, 0)


def filter_out_weak_phrases(texts, text_pad_masks, sims, max_phrases):
    if ((1 - text_pad_masks.to(int)).sum() <= max_phrases).all():
        return texts, text_pad_masks, sims

    new_texts, new_sims = [], []
    for i, sim in enumerate(sims):
        if sim.shape[0] > max_phrases:
            peak_sim = sim.max(dim=1).values
            top_indices = torch.sort(torch.topk(peak_sim, max_phrases).indices).values
            new_texts.append(texts[i][top_indices])
            new_sims.append(sim[top_indices])
        else:
            new_texts.append(texts[i])
            new_sims.append(sim)

    padded_texts, padded_masks = [], []
    for i, new_text in enumerate(new_texts):
        new_len = len(new_text)

        padded_text = F.pad(new_text, [0, 0, 0, max_phrases - new_len])
        padded_texts.append(padded_text)

        pad_mask = torch.ones(max_phrases).to(bool)
        pad_mask[:new_len] = False
        padded_masks.append(pad_mask)

    return torch.stack(padded_texts, 0), torch.stack(padded_masks, 0), new_sims


def load_last_checkpoint(
    name,
    model,
    device="cuda",
    remove_name_preffix=None,
    remove_name_postfix=None,
    ignore=None,
    models_path=None,
):
    models_path = models_path if models_path is not None else PROJECT_PATH
    weight_files = os.path.join(models_path, "weights", name, "weights-epoch=*.ckpt")
    latest_file = max(glob(weight_files), key=os.path.getctime)
    state_dict = torch.load(latest_file, map_location=device)["state_dict"]
    print(f"Loading checkpoint at {latest_file}")

    # adjust names in state dict
    new_keys = list(state_dict.keys())
    if remove_name_preffix:
        new_keys = [k[len(remove_name_preffix) :] for k in new_keys]
    if remove_name_postfix:
        new_keys = [k[: -len(remove_name_preffix)] for k in new_keys]

    # load state dict with new keys
    new_state_dict = dict(zip(new_keys, state_dict.values()))
    if ignore is not None:
        for ignore_key in ignore:
            new_state_dict.pop(ignore_key)

    model.load_state_dict(new_state_dict, strict=(ignore is None))
    return None


def get_decoder():
    features_dim = 768 if CONFIG.DATASET.FTYPE == "UniVL" else 512
    if CONFIG.MODEL.NAME == "TransformerQueryDecoder":
        model = VideoTransformerQueryDecoder(
            d_model=features_dim,
            num_layers=CONFIG.MODEL.NUM_LAYERS,
            num_heads=CONFIG.MODEL.NUM_HEADS,
            num_queries=CONFIG.MODEL.NUM_QUERIES,
            hidden_dim=CONFIG.MODEL.HIDDEN_DIM,
            d_output=features_dim,
            hidden_dropout=CONFIG.MODEL.HIDDEN_DROPOUT,
            output_dropout=CONFIG.MODEL.OUTPUT_DROPOUT,
            use_feature_pos_enc=CONFIG.MODEL.USE_POSITIONAL_ENCODING,
        )
    else:
        assert False, f"No such model as {CONFIG.MODEL.NAME}"
    return model


def get_optimizer(params, lr, global_step, epoch_len):
    # Define optimizer
    if CONFIG.TRAIN.OPTIMIZER == "Adam":
        opt_fn = torch.optim.Adam
    elif CONFIG.TRAIN.OPTIMIZER == "AdamW":
        opt_fn = torch.optim.AdamW
    else:
        assert False, f"No such optimizer {CONFIG.TRAIN.OPTIMIZER}"

    optimizer = opt_fn(params, lr=lr, weight_decay=CONFIG.TRAIN.WEIGHT_DECAY)

    # Define LR scheduler
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG.TRAIN.WARMUP_EPOCHS * epoch_len,
        num_training_steps=CONFIG.TRAIN.NUM_EPOCHS * epoch_len,
        last_epoch=global_step - 1,
    )
    scheduler = {"scheduler": scheduler, "interval": "step"}
    return optimizer, scheduler
