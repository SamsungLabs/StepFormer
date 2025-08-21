import pytorch_lightning as pl

from losses.losses import (
    compute_align_corresp_loss,
    compute_contrastive_loss,
    compute_video_step_seg_loss,
    compute_step_attn_reg_loss,
)
from losses.video_losses import compute_video_align_corresp_loss, compute_intra_video_loss, compute_npair_video_reg_loss
from config import CONFIG


class LossModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # define all possible losses
        self.define_losses()

        # collecting active losses and their metrics
        self.active_losses, self.active_metrics = [], ["total_loss"]
        for loss_name, loss in self.losses.items():
            if loss.mult > 0:
                self.active_losses.append(loss_name)
                self.active_metrics.extend(loss.metric_names)

    def define_losses(self):
        self.losses = dict()

        # Phrase losses
        self.losses["phrase_align_loss"] = AlignCorrespLoss(
            mult=CONFIG.LOSS.PHRASE_ALIGN_MULT,
            keep_percentile=CONFIG.LOSS.KEEP_PERCENTILE,
            top_band=CONFIG.LOSS.TOP_BAND,
            gamma_zx=CONFIG.LOSS.GAMMA_ZX,
            l2_normalize=CONFIG.LOSS.L2_NORM,
            one_to_many=CONFIG.LOSS.PHRASE_ALIGN_CONTRAST_MANY,
            metrics_preffix="phrase",
        )

        self.losses["phrase_contrastive_loss"] = ContrastiveLoss(
            mult=CONFIG.LOSS.PHRASE_CONTRASTIVE_MULT,
            gamma_zx=CONFIG.LOSS.GAMMA_ZX,
            l2_normalize=CONFIG.LOSS.L2_NORM,
            set_to_set=CONFIG.LOSS.CONTRAST_SETS,
            contrast_half=CONFIG.LOSS.CONTRAST_HALF,
            metrics_preffix="phrase",
        )

        # Video losses
        self.losses["video_align_loss"] = VideoAlignLoss(
            mult=CONFIG.LOSS.VIDEO_ALIGN_MULT,
            keep_percentile=CONFIG.LOSS.VIDEO_KEEP_PERCENTILE,
            gamma_zx=CONFIG.LOSS.GAMMA_ZX,
            l2_normalize=CONFIG.LOSS.L2_NORM,
            filter_steps=CONFIG.LOSS.VIDEO_ALIGN_FILTER_STEPS,
            contrast_frames=CONFIG.LOSS.VIDEO_ALIGN_CONTRAST_FRAMES,
            metrics_preffix="video",
        )

        self.losses["video_contrastive_loss"] = ContrastiveLoss(
            mult=CONFIG.LOSS.VIDEO_CONTRASTIVE_MULT,
            gamma_zx=CONFIG.LOSS.GAMMA_ZX,
            l2_normalize=CONFIG.LOSS.L2_NORM,
            set_to_set=CONFIG.LOSS.CONTRAST_SETS,
            contrast_half=CONFIG.LOSS.CONTRAST_HALF,
            metrics_preffix="video",
        )

        self.losses["video_step_seg_loss"] = VideoStepSegLoss(
            mult=CONFIG.LOSS.VIDEO_STEP_SEG_MULT,
            keep_percentile=CONFIG.LOSS.KEEP_PERCENTILE,
            top_band=CONFIG.LOSS.TOP_BAND,
            gamma_zx=CONFIG.LOSS.GAMMA_ZX,
            l2_normalize=CONFIG.LOSS.L2_NORM,
            contrast_level=CONFIG.LOSS.SEG_CONTRAST_LEVEL,
            contrast_frames=CONFIG.LOSS.SEG_CONTRAST_FRAMES,
            contrast_steps=CONFIG.LOSS.SEG_CONTRAST_STEPS,
        )

        # Step Reg losses
        self.losses["step_attn_reg_loss"] = StepAttnRegLoss(
            mult=CONFIG.LOSS.STEP_ATTN_REG_MULT,
            gamma_zx=CONFIG.LOSS.GAMMA_ZX,
            l2_normalize=CONFIG.LOSS.L2_NORM,
        )

        if CONFIG.LOSS.VIDEO_INTRA_MULT > 0:
            # self.active_loss_names.append("video_intra_loss")
            raise NotImplementedError

        self.losses["video_npair_reg_loss"] = NpairRegLoss(
            mult=CONFIG.LOSS.VIDEO_NPAIR_REG_MULT,
            gamma_zx=CONFIG.LOSS.GAMMA_ZX,
            l2_normalize=CONFIG.LOSS.L2_NORM,
        )

    def forward(
        self,
        steps,
        video,
        text,
        video_mask,
        text_mask,
        filter_steps=None,
        given_loss_queue=None,
    ):
        loss_queue = [loss for loss in self.active_losses] if given_loss_queue is None else given_loss_queue
        total_loss, all_metrics, all_tensors = 0, dict(), dict()
        for loss_name in loss_queue:
            loss_fn = self.losses[loss_name]
            loss = 0

            # phrase loss on video_steps
            if loss_name == "phrase_align_loss":
                loss, metrics, tensors = loss_fn(steps, text, None, text_mask)
            if loss_name == "phrase_contrastive_loss":
                loss, metrics, tensors = loss_fn(steps, text, None, text_mask)

            if loss_name == "video_align_loss":
                loss, metrics, tensors = loss_fn(steps, video, video_mask)
            if loss_name == "video_contrastive_loss":
                loss, metrics, tensors = loss_fn(steps, video, None, video_mask)
            if loss_name == "video_step_seg_loss":
                loss, metrics, tensors = loss_fn(steps, video, text, video_mask, text_mask)

            if loss_name == "step_attn_reg_loss":
                loss, metrics, tensors = loss_fn(steps, video, video_mask)
            if loss_name == "video_npair_reg_loss":
                loss, metrics, tensors = loss_fn(steps, video, video_mask)

            total_loss += loss
            all_metrics.update(metrics)
            all_tensors.update(tensors)
        all_metrics["total_loss"] = total_loss.item()
        return total_loss, all_metrics, all_tensors


class AlignCorrespLoss(pl.LightningModule):
    def __init__(self, mult, keep_percentile, top_band, gamma_zx, l2_normalize, one_to_many, metrics_preffix=""):
        super().__init__()
        self.mult = mult
        self.kp = keep_percentile
        self.tb = top_band
        self.gamma = gamma_zx
        self.l2_norm = l2_normalize
        self.one_to_many = one_to_many
        self.metric_names = ["_".join([metrics_preffix, n]) for n in ["align_loss"]]
        self.tensor_names = ["_".join([metrics_preffix, n]) for n in ["sims", "corresp_mats"]]

    def forward(self, steps, text, step_mask, text_mask, given_sims=None, given_corresp=None):
        loss, sims, corresp_mats = compute_align_corresp_loss(
            steps,
            text,
            step_mask,
            text_mask,
            keep_percentile=self.kp,
            top_band=self.tb,
            gamma_zx=self.gamma,
            l2_normalize=self.l2_norm,
            given_sims=given_sims,
            given_correspondences=given_corresp,
            one_to_many=self.one_to_many,
        )
        mult_loss = loss * self.mult
        metrics = {self.metric_names[0]: loss.item()}
        tensors = dict(zip(self.tensor_names, [sims, corresp_mats]))
        return mult_loss, metrics, tensors


class ContrastiveLoss(pl.LightningModule):
    def __init__(self, mult, gamma_zx, l2_normalize, set_to_set, contrast_half, metrics_preffix):
        super().__init__()
        self.mult = mult
        self.gamma = gamma_zx
        self.l2_norm = l2_normalize
        self.set_to_set = set_to_set
        self.contrast_half = contrast_half
        self.metric_names = ["_".join([metrics_preffix, n]) for n in ["contrastive_loss"]]

    def forward(self, steps, text, step_mask, text_mask):
        loss = compute_contrastive_loss(
            steps,
            text,
            step_mask,
            text_mask,
            gamma_zx=self.gamma,
            l2_normalize=self.l2_norm,
            set_to_set=self.set_to_set,
            contrast_half=self.contrast_half,
        )
        return loss * self.mult, {self.metric_names[0]: loss.item()}, {}


class VideoAlignLoss(pl.LightningModule):
    def __init__(
        self, mult, keep_percentile, gamma_zx, l2_normalize, filter_steps, contrast_frames, metrics_preffix=""
    ):
        super().__init__()
        self.mult = mult
        self.kp = keep_percentile
        self.gamma = gamma_zx
        self.l2_norm = l2_normalize
        self.filter_steps = filter_steps
        self.contrast_frames = contrast_frames
        self.metric_names = ["_".join([metrics_preffix, n]) for n in ["align_loss"]]
        self.tensor_names = ["_".join([metrics_preffix, n]) for n in ["sims", "corresp_mats"]]

    def forward(self, steps, video, video_mask, phrase_step_corresp_mats=None):
        if self.filter_steps:
            assert phrase_step_corresp_mats is not None, "Need phrase_step_corresp_mats to filter steps"
            # in case aligning only the steps, matched to some phrases
            masks = [cm.to(bool).any(1) for cm in phrase_step_corresp_mats]
            filtered_steps = [steps[i][mask] for i, mask in enumerate(masks)]
        else:
            filtered_steps = steps

        loss, sims, corresp_mats = compute_video_align_corresp_loss(
            filtered_steps,
            video,
            None,
            video_mask,
            keep_percentile=self.kp,
            gamma_zx=self.gamma,
            l2_normalize=self.l2_norm,
            contrast_frames=self.contrast_frames,
        )

        mult_loss = loss * self.mult
        metrics = {self.metric_names[0]: loss.item()}
        tensors = dict(zip(self.tensor_names, [sims, corresp_mats]))
        return mult_loss, metrics, tensors


class VideoStepSegLoss(pl.LightningModule):
    def __init__(
        self, mult, keep_percentile, top_band, gamma_zx, l2_normalize, contrast_level, contrast_frames, contrast_steps
    ):
        super().__init__()
        self.mult = mult
        self.kp = keep_percentile
        self.tb = top_band
        self.gamma = gamma_zx
        self.l2_norm = l2_normalize
        self.contrast_level = contrast_level
        self.contrast_frames = contrast_frames
        self.contrast_steps = contrast_steps
        self.metric_names = ["video_seg_loss"]
        self.tensor_names = ["video_seg_sim", "video_seg_corresp_mats"]

    def forward(self, steps, video, text, video_mask, text_mask, phrase_step_corresp_mats=None):
        loss, sims, corresp_mats = compute_video_step_seg_loss(
            video,
            text,
            steps,
            video_mask,
            text_mask,
            keep_percentile=self.kp,
            top_band=self.tb,
            gamma_zx=self.gamma,
            l2_normalize=self.l2_norm,
            contrast_level=self.contrast_level,
            contrast_frames=self.contrast_frames,
            contrast_steps=self.contrast_steps,
        )

        mult_loss = loss * self.mult
        metrics = {self.metric_names[0]: loss.item()}
        tensors = dict(zip(self.tensor_names, [sims, corresp_mats]))
        return mult_loss, metrics, tensors


class StepAttnRegLoss(pl.LightningModule):
    def __init__(self, mult, gamma_zx, l2_normalize):
        super().__init__()
        self.mult = mult
        self.gamma = gamma_zx
        self.l2_norm = l2_normalize
        self.metric_names = ["step_attn_reg_loss"]

    def forward(self, steps, video, video_mask):
        loss = compute_step_attn_reg_loss(
            steps,
            video,
            video_mask,
            gamma_zx=self.gamma,
            l2_normalize=self.l2_norm,
        )
        return loss * self.mult, {self.metric_names[0]: loss.item()}, {}


class NpairRegLoss(pl.LightningModule):
    def __init__(self, mult, gamma_zx, l2_normalize):
        super().__init__()
        self.mult = mult
        self.gamma = gamma_zx
        self.l2_norm = l2_normalize
        self.metric_names = ["video_npair_reg_loss"]

    def forward(self, steps, video, video_mask):
        loss = compute_npair_video_reg_loss(
            steps,
            video,
            None,
            video_mask,
            gamma_zx=self.gamma,
            l2_normalize=self.l2_norm,
        )
        return loss * self.mult, {self.metric_names[0]: loss.item()}, {}
