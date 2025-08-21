import os
import torch
import argparse
import random
import numpy as np
import pandas as pd
from os.path import join as opj
from tqdm import tqdm

from models.model_utils import load_last_checkpoint, get_decoder
from data.data_utils import get_dataset_info
from data.crosstask import CrossTaskModule
from data.procel import ProceLModule
from data.coin import COINModule

from eval.global_eval import eval_unsup_segmentation, step_video_segmentation
from config import CONFIG
from paths import PROJECT_PATH


# Enabling reproducibility
random.seed(2021)
np.random.seed(2021)
torch.manual_seed(2021)


def evaluate(args, data, model):
    name2task, task2steps = get_dataset_info(args.dataset, mode="val")

    # Step 1, compute step alignment on CrossTask
    dev = list(model.parameters())[0].device if model is not None else None
    all_pred_steps, all_vids, num_steps, all_names = {}, {}, {}, {}
    for batch in data.val_dataloader():
        # Step 1: get features, labels and global task info
        name = batch["name"][0]
        name = name[0] if type(name) == tuple else name
        if name not in name2task:
            continue
        task_name = name2task[name]
        video_features = batch["video_features"][0][~batch["video_pad_mask"][0]]

        # Step 2: get predicted steps using our trained model
        with torch.no_grad():
            pred_steps = model(video_features[None, ...].to(dev)).to("cpu")[0]

        # Step 3: accumulate features and labels on task-based level to compare to CVPR baseline
        if task_name not in all_vids.keys():
            all_vids[task_name] = []
            all_names[task_name] = []
            all_pred_steps[task_name] = []
            num_steps[task_name] = len(task2steps[task_name])

        all_vids[task_name].append(video_features.detach())
        all_pred_steps[task_name].append(pred_steps)
        all_names[task_name].append(name)

    all_results = []
    for task in tqdm(all_vids):
        # Step 4: get pre-task segmentation
        pred_labels, _ = step_video_segmentation(
            all_vids[task],
            all_pred_steps[task],
            K=num_steps[task],
            bg_ratio=args.bg_ratio,
            keep_percentile=CONFIG.EVAL.KEEP_PERCENTILE,
            dist_metric="cosine",
        )

        # dump prediction into a file
        pred_root = opj(PROJECT_PATH, "preds")
        pred_dir = opj(pred_root, task + "-val")
        os.makedirs(pred_dir, exist_ok=True)
        for pred_id, pred in enumerate(pred_labels):
            name = all_names[task][pred_id]
            pred_name = name if args.dataset in ["COIN", "ProceL"] else task + "-val" + "_" + name
            pred_path = opj(pred_dir, pred_name + ".txt")
            np.savetxt(pred_path, pred.astype(int), fmt="%i")

        # Step 5: get pre-task validation metrics
        results = eval_unsup_segmentation(
            args.dataset, task, pred_root, n_frames=16, framerate=16, verbose=args.verbose
        )
        all_results.append(results)

    avg_metrics = [np.mean(r) for r in list(zip(*all_results))]
    print(
        str(args.dataset) + " Unsupervised Segmentation: Precision: {:.2%}, Recall: {:.2%}, F1-score: {:.2%}, "
        "MoF: {:.2%}, MoF-bg: {:.2%}".format(*avg_metrics)
    )


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="howto_test", help="name of the model (experiment) we are testing")
    parser.add_argument("--dataset", type=str, default="CrossTask", choices=["ProceL", "CrossTask", "COIN"], help="name of the dataset")
    parser.add_argument("--bg_ratio", type=float, default=0.4, help="background threshold for the global clustering")
    parser.add_argument("--verbose", type=int, default=0, help="print everything")
    parser.add_argument("--override", nargs=argparse.REMAINDER, help='override train/val keys in CONFIG')
    args = parser.parse_args()
    # fmt: on

    # setup from config
    output_folder = os.path.join(PROJECT_PATH, "outputs", args.name)
    CONFIG.load(os.path.join(output_folder, "config.yml"))
    CONFIG.override_with_args(args.override)

    # prep data
    if args.dataset == "CrossTask":
        data = CrossTaskModule(whiten=CONFIG.DATASET.WHITEN)
    elif args.dataset == "ProceL":
        data = ProceLModule(whiten=CONFIG.DATASET.WHITEN)
    elif args.dataset == "COIN":
        data = COINModule(whiten=CONFIG.DATASET.WHITEN)
    else:
        print("UNKNOWN DATASET")

    # prep model
    model = get_decoder()
    load_last_checkpoint(args.name, model, remove_name_preffix="model.")
    model.eval()

    # evaluate
    evaluate(args, data, model)


if __name__ == "__main__":
    main()
