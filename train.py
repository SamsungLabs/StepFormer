import os
import torch
import argparse
import random
import numpy as np
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
from glob import glob
import wandb

from models.model_utils import load_last_checkpoint, get_decoder, get_optimizer
from losses.loss_module import LossModule

from data.tar_loader import TarDataModule as DataModule
from data.crosstask import CrossTaskModule
from data.overfit import OverfitDataModule
from utils import get_logger, log_info
from eval.framewise_eval import evaluate_predicted_steps_zeroshot, evaluate_predicted_steps_unsupervised
from dp.visualization import plot_step_to_video_alignment, plot_similarities
from config import CONFIG
from paths import PROJECT_PATH

# Enabling reproducibility
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)

# Setting up arguments
# fmt: off
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="", help="name of the experiment")
parser.add_argument("--short_name", type=str, default="", help="short name of the experiment for visualization in wandb")
parser.add_argument("--overfit", type=int, default=0, help="overfit n batches")
parser.add_argument("--dbg", action="store_true", help="puts the dataset in local debug mode")
parser.add_argument("--resume_training", action="store_true", help="continue from last checkpoint")
parser.add_argument("--init_from", type=str, default="", help="name of the model to init the current model from")
parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus used for training")
parser.add_argument("--ml_logger", default='wandb', choices=['wandb', 'tensorboard'], help="puts the dataset in local debug mode")
parser.add_argument("--parallelism", type=str, default="horovod", help="number of gpus used for training")
parser.add_argument("--model_name", type=str, default="TransformerQueryDecoder", choices=["TransformerQueryDecoder"], help="name of the dataset we are encoding")
parser.add_argument("--dataset", type=str, default="HowTo100M", choices=["HowTo100M", "CrossTask"], help="name of the dataset we are encoding")
parser.add_argument("--config_path", type=str, default="./conf/models/", help="path to config file")
parser.add_argument("--override", nargs=argparse.REMAINDER)
args = parser.parse_args()
# fmt: on

# setup output directory and logging
OUTPUT_PATH = os.path.join(PROJECT_PATH, "outputs", args.name)
os.makedirs(OUTPUT_PATH) if not os.path.isdir(OUTPUT_PATH) else None
logger = get_logger(log_file=os.path.join(OUTPUT_PATH, "train.log"), to_console=False)


class TrainModule(pl.LightningModule):
    def __init__(self, model, data, name=None):
        super(TrainModule, self).__init__()
        self.name = name
        self.model = model
        self.data = data
        self.loss = LossModule()

        self.train_metrics = torch.nn.ModuleDict({name: torchmetrics.MeanMetric() for name in self.loss.active_metrics})
        self.val_metrics = torch.nn.ModuleDict({name: torchmetrics.MeanMetric() for name in self.loss.active_metrics})

    def configure_optimizers(self):
        # set lr depending on the batch_size
        lr = CONFIG.TRAIN.LR * CONFIG.TRAIN.BATCH_SIZE / 8
        lr = lr * max(1, args.num_gpus) if args.parallelism in ["ddp", "horovod"] else lr

        optim_params = self.model.parameters()
        num_videos_per_folder = 1000  # was encoded that way
        data_len = int(len(self.data.train_folders) * num_videos_per_folder / args.num_gpus) * args.num_gpus
        epoch_len = int(data_len / (args.num_gpus * CONFIG.TRAIN.BATCH_SIZE))
        optimizer, scheduler = get_optimizer(optim_params, lr=lr, global_step=self.global_step, epoch_len=epoch_len)
        return [optimizer], [scheduler]

    def unpack_batch(self, batch):
        video, video_mask, text, text_mask = batch

        # filter out samples that have 0 frames or 0 phrases
        has_several_phrases = (text_mask != 1).any(1)
        has_several_frames = (video_mask != 1).any(1)
        good_mask = torch.logical_and(has_several_phrases, has_several_frames)
        video, text = video[good_mask], text[good_mask]
        video_mask, text_mask = video_mask[good_mask], text_mask[good_mask]

        steps = self.model(video, features_mask=video_mask)
        return steps, video, text, video_mask, text_mask

    def training_step(self, batch, batch_id):
        steps, video, text, video_mask, text_mask = self.unpack_batch(batch)
        batch_loss, metrics, tensors = self.loss(steps, video, text, video_mask, text_mask)

        for loss_name in self.loss.active_metrics:
            self.train_metrics[loss_name](metrics[loss_name])

        if (batch_id) % 100 == 0:
            # dumping scalar logs into tensorboard
            for loss_name in self.loss.active_metrics:
                self.log(f"train/{loss_name}", self.train_metrics[loss_name].compute())
                self.train_metrics[loss_name].reset()

        if (batch_id) % 1000 == 0:
            # logging correspondence and alignment statistics
            self.log_stats(
                steps, tensors.get("phrase_sims", None), tensors.get("phrase_corresp_mats", None), "PhraseAlign"
            )

            # visualize alignment for the 0-th element of the batch
            self.visualize_alignment(
                steps[[0]],
                video[[0]],
                video_mask[[0]],
                text[[0]],
                text_mask[[0]],
            )

        return batch_loss

    def validation_step(self, batch, batch_id):
        steps, video, text, video_mask, text_mask = self.unpack_batch(batch)
        _, metrics, _ = self.loss(steps, video, text, video_mask, text_mask)
        for loss_name in self.loss.active_metrics:
            self.val_metrics[loss_name](metrics[loss_name])

        for loss_name in self.loss.active_metrics:
            self.log(f"val/{loss_name}", metrics[loss_name])
        return None

    def log_histogram(self, name, hist):
        if args.ml_logger == "tensorboard":
            self.logger.experiment.add_histogram(name, hist, self.global_step)
        else:
            self.logger.experiment.log({name: wandb.Histogram(hist.detach().cpu())}, step=self.global_step)

    def log_image(self, name, image):
        if args.ml_logger == "tensorboard":
            image = image.transpose((2, 0, 1))
            self.logger.experiment.add_image(name, image, self.global_step)
        else:
            self.logger.experiment.log({name: wandb.Image(image)}, step=self.global_step)

    def log_stats(self, steps, sims, corresp_mats, log_name):
        # log steps self-similarity histograms
        if steps is not None:
            for b_id in range(len(steps)):
                norm_step = torch.nn.functional.normalize(steps[b_id], p=2, dim=1)
                steps_prod = norm_step @ norm_step.T
                if b_id == 0:
                    total_selfsim_vec = steps_prod.reshape(-1)
                else:
                    total_selfsim_vec = torch.cat([total_selfsim_vec, steps_prod.reshape(-1)], dim=0)
            self.log_histogram("step_cos_selfsim", total_selfsim_vec)

        # log steps to other sequence similarity
        if sims is not None:
            for b_id in range(len(sims)):
                flat_sim = sims[b_id].reshape(-1)
                sampled_vals = flat_sim[torch.randint(len(flat_sim), (2000,))]
                if b_id == 0:
                    total_sim_vec = sampled_vals
                else:
                    total_sim_vec = torch.cat([total_sim_vec, sampled_vals], dim=0)
            self.log_histogram(f"{log_name}/matching_similarities", total_sim_vec)

        if corresp_mats is not None:
            avg_matched_rows = sum([(cm == 1).any(1).to(int).sum() / cm.size(0) for cm in corresp_mats]) / len(sims)
            avg_matched_cols = sum([(cm == 1).any(0).to(int).sum() / cm.size(1) for cm in corresp_mats]) / len(sims)
            self.log(f"match_ratios/{log_name}/steps", avg_matched_rows)
            self.log(f"match_ratios/{log_name}/phrases", avg_matched_cols)

        # logging the full name in wandb
        if self.global_step == 0 and args.ml_logger == "wandb":
            self.logger.experiment.log({"full_name": wandb.Table(data=[[args.name]], columns=["full_name"])})

    def visualize_alignment(self, step, video, video_mask, text, text_mask):
        loss_queue = ["video_align_loss"]
        with torch.no_grad():
            _, _, tensors = self.loss(step, video, text, video_mask, text_mask, given_loss_queue=loss_queue)
        cm = tensors["video_corresp_mats"][0]
        matching_image = plot_step_to_video_alignment(cm)
        self.log_image("alignment/step2video", matching_image)

        sims = F.normalize(step[0], dim=1) @ F.normalize(video[0][~video_mask[0]], dim=1).T
        sim_image = plot_similarities(sims.detach().cpu().numpy(), color_offset=1)
        self.log_image("sim/step2video", sim_image)

    def training_epoch_end(self, training_step_outputs):
        # Perform validation via zero-shot localization on CrossTask and log the results

        # first, compute step alignment on CrossTask
        self.model.eval()
        dev = list(self.model.parameters())[0].device
        ct_metrics = {"Zero": {}, "Unsup": {}}
        N_total = 0
        for batch in self.data.crosstask.val_dataloader():
            video_features = batch["video_features"][0][~batch["video_pad_mask"][0]]
            phrase_features = batch["text_features"][0][~batch["text_pad_mask"][0]]
            json = batch["json"][0]

            with torch.no_grad():
                steps = self.model(video_features[None, ...].to(dev)).to("cpu")[0]

            for eval_type in ["Zero", "Unsup"]:
                if eval_type == "Zero":
                    metrics_dict, _ = evaluate_predicted_steps_zeroshot(
                        video_features,
                        phrase_features,
                        steps,
                        json,
                    )
                else:
                    metrics_dict, _ = evaluate_predicted_steps_unsupervised(
                        video_features,
                        phrase_features,
                        steps,
                        json,
                    )

                # accumulate metrics
                for key, metric in metrics_dict.items():
                    if key not in ct_metrics[eval_type]:
                        ct_metrics[eval_type][key] = 0
                    ct_metrics[eval_type][key] += ct_metrics[eval_type][key] + metric
            N_total += 1

        # then, log everything
        for mode, metrics_dict in zip(["Train", "Val"], [self.train_metrics, self.val_metrics]):
            for loss_name in self.loss.active_metrics:
                loss = metrics_dict[loss_name].compute()
                self.log(f"{mode}/{loss_name}".lower(), loss.item(), on_step=False, on_epoch=True)
                metrics_dict[loss_name].reset()
                log_message = f" {mode} | {loss_name}: {loss.item():.2f}"
                log_info(log_message, logger)
            log_info("--------------------------", logger)

        for eval_type, metrics_dict in ct_metrics.items():
            log_string = f"CrossTask {eval_type} | "
            for metric_name, metric_val in metrics_dict.items():
                report_val = metric_val * 100 / N_total
                self.log(f"CrossTask/{eval_type}/{metric_name}", report_val, on_step=False, on_epoch=True)
                log_string = log_string + f"{metric_name}: {report_val:.2f} , "
            log_info(log_string[:-2], logger)
        log_info(f" End of epoch {self.current_epoch} \n", logger)
        log_info("\n", logger)
        self.model.train()


def main():
    # setup from config
    CONFIG.setup(args.name, args.config_path, args.model_name, override_args=args.override)
    CONFIG.dump(os.path.join(OUTPUT_PATH, "config.yml"))
    if args.name == "" and args.short_name != "":
        args.name = args.short_name
    log_info(CONFIG.yaml_format(), logger)
    log_info(CONFIG.yaml_format())

    # prep data
    data = DataModule(dbg=args.dbg)
    data.crosstask = CrossTaskModule()
    if args.overfit:
        data = OverfitDataModule(data)
    # prep model
    model = get_decoder()
    if args.init_from != "":
        init_path = os.path.join(PROJECT_PATH, "weights")
        load_last_checkpoint(args.init_from, model, remove_name_preffix="model.", models_path=init_path)

    # prep train loop
    train_module = TrainModule(model, data)

    # prep callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/total_loss",
        dirpath=os.path.join(PROJECT_PATH, "weights", args.name),
        filename="weights-{epoch:02d}",
        save_top_k=1,
        mode="min",
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # prep tensorboard logger
    if args.ml_logger == "tensorboard":
        ml_logger = pl.loggers.TensorBoardLogger(os.path.join(PROJECT_PATH, "tb_logs"), args.name)
    else:
        short_name = args.short_name if args.short_name else None
        ml_logger = pl.loggers.WandbLogger(name=short_name, project="unsup-step-pred", config=CONFIG.dotdict_format())

    callbacks = [checkpoint_callback, lr_monitor]

    if args.resume_training:
        weight_files = os.path.join(PROJECT_PATH, "weights", args.name, "weights-epoch=*.ckpt")
        last_checkpoint = max(glob(weight_files), key=os.path.getctime)
    else:
        last_checkpoint = None

    # start training
    if args.num_gpus > 1:
        if args.parallelism == "horovod":
            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                callbacks=callbacks,
                max_epochs=CONFIG.TRAIN.NUM_EPOCHS,
                logger=ml_logger,
                resume_from_checkpoint=last_checkpoint,
                strategy=args.parallelism,
                num_sanity_val_steps=0,
                # gradient_clip_val=0.5,
                reload_dataloaders_every_n_epochs=1,
            )
        else:
            trainer = pl.Trainer(
                devices=args.num_gpus,
                accelerator="gpu",
                num_nodes=1,
                callbacks=callbacks,
                max_epochs=CONFIG.TRAIN.NUM_EPOCHS,
                logger=ml_logger,
                resume_from_checkpoint=last_checkpoint,
                strategy=args.parallelism,
                num_sanity_val_steps=0,
                # gradient_clip_val=0.5,
                reload_dataloaders_every_n_epochs=1,
            )
    else:
        trainer = pl.Trainer(
            gpus=1,
            callbacks=callbacks,
            max_epochs=CONFIG.TRAIN.NUM_EPOCHS,
            logger=ml_logger,
            resume_from_checkpoint=last_checkpoint,
            num_sanity_val_steps=0,
            # gradient_clip_val=0.5,
            reload_dataloaders_every_n_epochs=1,
        )
    trainer.fit(train_module, data)


if __name__ == "__main__":
    main()
