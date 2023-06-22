import os
import torch
import argparse
import random
import torch
import numpy as np
from tqdm import tqdm

from models.model_utils import load_last_checkpoint, get_decoder
from data.crosstask import CrossTaskModule
from data.procel import ProceLModule
from data.coin import COINModule
from config import CONFIG
from eval.framewise_eval import evaluate_predicted_steps_zeroshot
from paths import PROJECT_PATH


# Enabling reproducibility
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)


def evaluate(data, model):
    # first, compute step alignment on CrossTask
    model.eval()
    dev = list(model.parameters())[0].device

    all_metrics = dict()
    for N, batch in enumerate(tqdm(data.val_dataloader())):
        video_features = batch["video_features"][0][~batch["video_pad_mask"][0]]
        phrase_features = batch["text_features"][0][~batch["text_pad_mask"][0]]
        json = batch["json"][0]

        with torch.no_grad():
            steps = model(video_features[None, ...].to(dev)).to("cpu")[0]
        metrics_dict, _ = evaluate_predicted_steps_zeroshot(video_features, phrase_features, steps, json)

        # accumulate metrics
        for k, v in metrics_dict.items():
            all_metrics[k] = all_metrics.get(k, 0) + v

    log_string = "CrossTask Zero-Shot Step Localization | "
    for metric_name, metric_val in all_metrics.items():
        log_string = log_string + f"{metric_name}: {metric_val * 100 / (N + 1) :.1f} , "
    print(log_string[:-2])


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="howto_test", help="name of the experiment")
    parser.add_argument("--model_name", type=str, default="TransformerQueryDecoder", choices=["TransformerQueryDecoder"], help="name of the dataset we are encoding")
    parser.add_argument("--dataset", type=str, default="CrossTask", choices=["ProceL", "COIN", "CrossTask"], help="name of the dataset we are encoding")
    parser.add_argument("--override", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # fmt: on

    # setup from config
    output_folder = os.path.join(PROJECT_PATH, "outputs", args.name)
    CONFIG.load(os.path.join(output_folder, "config.yml"))
    CONFIG.override_with_args(args.override)

    # prep data
    if args.dataset == "CrossTask":
        data = CrossTaskModule()
    elif args.dataset == "ProceL":
        data = ProceLModule()
    elif args.dataset == "COIN":
        data = COINModule()
    else:
        print("UNKNOWN DATASET")

    # prep model
    model = get_decoder()
    load_last_checkpoint(args.name, model, remove_name_preffix="model.")
    model.eval()

    # evaluate
    evaluate(data, model)


if __name__ == "__main__":
    main()
